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
execution time: IAR + RelationalAnalysis = 7.79 + 28.33 = 36.12 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0162071, upper bound: 0.0162100

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 272
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 274
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 390
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 3264
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 345
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 377
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 2859
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 376
type: DSZ, layer: 1, pos: 3501
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 433
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 665
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 668
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2118
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2651
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2700
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 2860
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3364
type: DSZ, layer: 1, pos: 3374
type: DSZ, layer: 1, pos: 3381
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3529
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 272

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0161341, upper bound: 0.0162135
time: 8.54 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0162068, upper bound: 0.0161301
time: 20.59 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 29.21 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 29.21
Output dim: 1, lower bound: -0.0161341, upper bound: 0.0162135
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 29.21
Output dim: 1, lower bound: -0.0162068, upper bound: 0.0161301

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.8993219, -0.2406590, -0.8993219, -0.2406590, -0.3926476, 0.3926694
1: -0.0854008, 0.3246739, -0.0854008, 0.3246739, -0.0668271, 0.0668322
2: -2.2585669, -1.4722774, -2.2585669, -1.4722774, -0.2649102, 0.2649087
3: -2.8736234, -1.7064936, -2.8736234, -1.7064936, -0.4076709, 0.4076714
4: -3.5778260, -2.4280591, -3.5778260, -2.4280591, -0.4192420, 0.4192408
5: -2.5908875, -1.5438142, -2.5908875, -1.5438142, -0.2147288, 0.2147305
6: -2.5916283, -1.4587758, -2.5916283, -1.4587758, -0.1981455, 0.1981734
7: -2.8178353, -1.4403852, -2.8178353, -1.4403852, -0.5125239, 0.5125180
8: -0.0413489, 0.2033695, -0.0413489, 0.2033695, -0.1476325, 0.1476324
9: -0.8593886, -0.4630676, -0.8593886, -0.4630676, -0.2108963, 0.2109005

Time for backsubstitution: 5.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 274
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 390
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 3264
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 345
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 377
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 2859
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 376
type: DSZ, layer: 1, pos: 3501
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 433
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 665
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 668
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2118
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2651
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2700
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 2860
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3364
type: DSZ, layer: 1, pos: 3374
type: DSZ, layer: 1, pos: 3381
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3529
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 553

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0160977, upper bound: 0.0162144
time: 47.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0161336, upper bound: 0.0161717
time: 106.22 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.8993219, -0.2406590, -0.8993219, -0.2406590, -0.3926694, 0.3926476
1: -0.0854008, 0.3246739, -0.0854008, 0.3246739, -0.0668322, 0.0668271
2: -2.2585669, -1.4722774, -2.2585669, -1.4722774, -0.2649087, 0.2649101
3: -2.8736234, -1.7064936, -2.8736234, -1.7064936, -0.4076713, 0.4076709
4: -3.5778260, -2.4280591, -3.5778260, -2.4280591, -0.4192408, 0.4192420
5: -2.5908875, -1.5438142, -2.5908875, -1.5438142, -0.2147305, 0.2147288
6: -2.5916283, -1.4587758, -2.5916283, -1.4587758, -0.1981734, 0.1981455
7: -2.8178353, -1.4403852, -2.8178353, -1.4403852, -0.5125180, 0.5125239
8: -0.0413489, 0.2033695, -0.0413489, 0.2033695, -0.1476324, 0.1476325
9: -0.8593886, -0.4630676, -0.8593886, -0.4630676, -0.2109005, 0.2108963

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 274
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 390
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 3264
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 345
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 377
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 2859
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 376
type: DSZ, layer: 1, pos: 3501
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 433
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 665
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 668
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2118
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2651
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2700
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 2860
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3364
type: DSZ, layer: 1, pos: 3374
type: DSZ, layer: 1, pos: 3381
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3529
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 553

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0161704, upper bound: 0.0161431
time: 4.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0162063, upper bound: 0.0160983
time: 27.43 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 38.00 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 38.00
Output dim: 1, lower bound: -0.0160977, upper bound: 0.0162144
DS_DSZ1_DSZ2, status: Status.VERIFIED, split count: 2, time: 38.00
Output dim: 1, lower bound: -0.0161336, upper bound: 0.0161717
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 38.00
Output dim: 1, lower bound: -0.0161704, upper bound: 0.0161431
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 38.00
Output dim: 1, lower bound: -0.0162063, upper bound: 0.0160983

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.8993219, -0.2406590, -0.8993219, -0.2406590, -0.3924840, 0.3925006
1: -0.0854008, 0.3246739, -0.0854008, 0.3246739, -0.0660980, 0.0661297
2: -2.2585669, -1.4722774, -2.2585669, -1.4722774, -0.2638573, 0.2638198
3: -2.8736234, -1.7064936, -2.8736234, -1.7064936, -0.4067612, 0.4067433
4: -3.5778260, -2.4280591, -3.5778260, -2.4280591, -0.4186569, 0.4186720
5: -2.5908875, -1.5438142, -2.5908875, -1.5438142, -0.2139595, 0.2139470
6: -2.5916283, -1.4587758, -2.5916283, -1.4587758, -0.1978009, 0.1978314
7: -2.8178353, -1.4403852, -2.8178353, -1.4403852, -0.5118551, 0.5118386
8: -0.0413489, 0.2033695, -0.0413489, 0.2033695, -0.1474327, 0.1474394
9: -0.8593886, -0.4630676, -0.8593886, -0.4630676, -0.2108941, 0.2108984

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 274
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 390
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 3264
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 345
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 377
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 2859
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 376
type: DSZ, layer: 1, pos: 3501
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 433
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 665
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 668
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2118
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2651
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2700
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 2860
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3364
type: DSZ, layer: 1, pos: 3374
type: DSZ, layer: 1, pos: 3381
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3529
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 274

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0160455, upper bound: 0.0162082
time: 44.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0160978, upper bound: 0.0161592
time: 21.00 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.8993219, -0.2406590, -0.8993219, -0.2406590, -0.3925006, 0.3924841
1: -0.0854008, 0.3246739, -0.0854008, 0.3246739, -0.0661297, 0.0660980
2: -2.2585669, -1.4722774, -2.2585669, -1.4722774, -0.2638198, 0.2638572
3: -2.8736234, -1.7064936, -2.8736234, -1.7064936, -0.4067433, 0.4067612
4: -3.5778260, -2.4280591, -3.5778260, -2.4280591, -0.4186720, 0.4186569
5: -2.5908875, -1.5438142, -2.5908875, -1.5438142, -0.2139470, 0.2139594
6: -2.5916283, -1.4587758, -2.5916283, -1.4587758, -0.1978313, 0.1978009
7: -2.8178353, -1.4403852, -2.8178353, -1.4403852, -0.5118386, 0.5118551
8: -0.0413489, 0.2033695, -0.0413489, 0.2033695, -0.1474394, 0.1474327
9: -0.8593886, -0.4630676, -0.8593886, -0.4630676, -0.2108984, 0.2108940

Time for backsubstitution: 6.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 274
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 390
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 3264
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 345
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 377
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 2859
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 376
type: DSZ, layer: 1, pos: 3501
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 433
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 665
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 668
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2118
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2651
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2700
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 2860
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3364
type: DSZ, layer: 1, pos: 3374
type: DSZ, layer: 1, pos: 3381
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3529
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 274

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0161541, upper bound: 0.0161007
time: 83.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0162064, upper bound: 0.0160531
time: 6.55 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 95.86 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 95.86
Output dim: 1, lower bound: -0.0160455, upper bound: 0.0162082
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 95.86
Output dim: 1, lower bound: -0.0160978, upper bound: 0.0161592
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 95.86
Output dim: 1, lower bound: -0.0161541, upper bound: 0.0161007
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 95.86
Output dim: 1, lower bound: -0.0162064, upper bound: 0.0160531

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.8993219, -0.2406590, -0.8993219, -0.2406590, -0.3924811, 0.3924945
1: -0.0854008, 0.3246739, -0.0854008, 0.3246739, -0.0660846, 0.0661252
2: -2.2585669, -1.4722774, -2.2585669, -1.4722774, -0.2638571, 0.2638452
3: -2.8736234, -1.7064936, -2.8736234, -1.7064936, -0.4067359, 0.4067224
4: -3.5778260, -2.4280591, -3.5778260, -2.4280591, -0.4186515, 0.4186640
5: -2.5908875, -1.5438142, -2.5908875, -1.5438142, -0.2138718, 0.2138684
6: -2.5916283, -1.4587758, -2.5916283, -1.4587758, -0.1977611, 0.1978173
7: -2.8178353, -1.4403852, -2.8178353, -1.4403852, -0.5117598, 0.5116109
8: -0.0413489, 0.2033695, -0.0413489, 0.2033695, -0.1473902, 0.1474172
9: -0.8593886, -0.4630676, -0.8593886, -0.4630676, -0.2108708, 0.2108794

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 390
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 3264
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 345
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 377
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 2859
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 376
type: DSZ, layer: 1, pos: 3501
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 433
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 665
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 668
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2118
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2651
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2700
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 2860
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3364
type: DSZ, layer: 1, pos: 3374
type: DSZ, layer: 1, pos: 3381
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3529
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3492

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0159651, upper bound: 0.0162159
time: 7.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0160434, upper bound: 0.0161171
time: 54.87 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.8993219, -0.2406590, -0.8993219, -0.2406590, -0.3924944, 0.3924811
1: -0.0854008, 0.3246739, -0.0854008, 0.3246739, -0.0661252, 0.0660846
2: -2.2585669, -1.4722774, -2.2585669, -1.4722774, -0.2638452, 0.2638571
3: -2.8736234, -1.7064936, -2.8736234, -1.7064936, -0.4067223, 0.4067360
4: -3.5778260, -2.4280591, -3.5778260, -2.4280591, -0.4186640, 0.4186515
5: -2.5908875, -1.5438142, -2.5908875, -1.5438142, -0.2138684, 0.2138718
6: -2.5916283, -1.4587758, -2.5916283, -1.4587758, -0.1978173, 0.1977611
7: -2.8178353, -1.4403852, -2.8178353, -1.4403852, -0.5116109, 0.5117598
8: -0.0413489, 0.2033695, -0.0413489, 0.2033695, -0.1474172, 0.1473902
9: -0.8593886, -0.4630676, -0.8593886, -0.4630676, -0.2108794, 0.2108708

Time for backsubstitution: 6.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 390
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 3264
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 345
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 377
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 2859
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 376
type: DSZ, layer: 1, pos: 3501
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 433
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 665
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 668
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2118
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2651
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2700
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 2860
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3364
type: DSZ, layer: 1, pos: 3374
type: DSZ, layer: 1, pos: 3381
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3529
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3492

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0161091, upper bound: 0.0160373
time: 20.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0162067, upper bound: 0.0159655
time: 120.46 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 146.69 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 146.69
Output dim: 1, lower bound: -0.0159651, upper bound: 0.0162159
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 146.69
Output dim: 1, lower bound: -0.0160434, upper bound: 0.0161171
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 146.69
Output dim: 1, lower bound: -0.0161091, upper bound: 0.0160373
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 146.69
Output dim: 1, lower bound: -0.0162067, upper bound: 0.0159655

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.8993219, -0.2406590, -0.8993219, -0.2406590, -0.3922951, 0.3923017
1: -0.0854008, 0.3246739, -0.0854008, 0.3246739, -0.0652331, 0.0653429
2: -2.2585669, -1.4722774, -2.2585669, -1.4722774, -0.2627219, 0.2626156
3: -2.8736234, -1.7064936, -2.8736234, -1.7064936, -0.4057857, 0.4056932
4: -3.5778260, -2.4280591, -3.5778260, -2.4280591, -0.4179849, 0.4180483
5: -2.5908875, -1.5438142, -2.5908875, -1.5438142, -0.2130310, 0.2129576
6: -2.5916283, -1.4587758, -2.5916283, -1.4587758, -0.1973956, 0.1974200
7: -2.8178353, -1.4403852, -2.8178353, -1.4403852, -0.5110820, 0.5108764
8: -0.0413489, 0.2033695, -0.0413489, 0.2033695, -0.1471652, 0.1472095
9: -0.8593886, -0.4630676, -0.8593886, -0.4630676, -0.2108693, 0.2108780

Time for backsubstitution: 6.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 390
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 3264
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 345
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 377
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 2859
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 376
type: DSZ, layer: 1, pos: 3501
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 433
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 665
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 668
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2118
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2651
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2700
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 2860
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3364
type: DSZ, layer: 1, pos: 3374
type: DSZ, layer: 1, pos: 3381
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3529
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 390

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0159637, upper bound: 0.0161703
time: 4.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0159219, upper bound: 0.0162108
time: 5.11 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.8993219, -0.2406590, -0.8993219, -0.2406590, -0.3923017, 0.3922950
1: -0.0854008, 0.3246739, -0.0854008, 0.3246739, -0.0653429, 0.0652331
2: -2.2585669, -1.4722774, -2.2585669, -1.4722774, -0.2626156, 0.2627219
3: -2.8736234, -1.7064936, -2.8736234, -1.7064936, -0.4056932, 0.4057857
4: -3.5778260, -2.4280591, -3.5778260, -2.4280591, -0.4180483, 0.4179849
5: -2.5908875, -1.5438142, -2.5908875, -1.5438142, -0.2129576, 0.2130310
6: -2.5916283, -1.4587758, -2.5916283, -1.4587758, -0.1974201, 0.1973956
7: -2.8178353, -1.4403852, -2.8178353, -1.4403852, -0.5108764, 0.5110820
8: -0.0413489, 0.2033695, -0.0413489, 0.2033695, -0.1472095, 0.1471652
9: -0.8593886, -0.4630676, -0.8593886, -0.4630676, -0.2108780, 0.2108693

Time for backsubstitution: 6.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 390
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 3264
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 345
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 377
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 2859
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 376
type: DSZ, layer: 1, pos: 3501
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 433
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 665
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 668
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2118
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2651
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2700
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 2860
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3364
type: DSZ, layer: 1, pos: 3374
type: DSZ, layer: 1, pos: 3381
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3529
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 390

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0162054, upper bound: 0.0159291
time: 14.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0161636, upper bound: 0.0159297
time: 58.57 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 78.89 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 78.89
Output dim: 1, lower bound: -0.0159637, upper bound: 0.0161703
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 78.89
Output dim: 1, lower bound: -0.0159219, upper bound: 0.0162108
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 78.89
Output dim: 1, lower bound: -0.0162054, upper bound: 0.0159291
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 78.89
Output dim: 1, lower bound: -0.0161636, upper bound: 0.0159297

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.8993219, -0.2406590, -0.8993219, -0.2406590, -0.3866106, 0.3864202
1: -0.0854008, 0.3246739, -0.0854008, 0.3246739, -0.0647171, 0.0648450
2: -2.2585669, -1.4722774, -2.2585669, -1.4722774, -0.2642085, 0.2641876
3: -2.8736234, -1.7064936, -2.8736234, -1.7064936, -0.4017068, 0.4017510
4: -3.5778260, -2.4280591, -3.5778260, -2.4280591, -0.4157205, 0.4159189
5: -2.5908875, -1.5438142, -2.5908875, -1.5438142, -0.2096542, 0.2096939
6: -2.5916283, -1.4587758, -2.5916283, -1.4587758, -0.1932245, 0.1933999
7: -2.8178353, -1.4403852, -2.8178353, -1.4403852, -0.5059665, 0.5062617
8: -0.0413489, 0.2033695, -0.0413489, 0.2033695, -0.1476099, 0.1476314
9: -0.8593886, -0.4630676, -0.8593886, -0.4630676, -0.2098081, 0.2098527

Time for backsubstitution: 6.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 3264
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 345
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 377
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 2859
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 376
type: DSZ, layer: 1, pos: 3501
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 433
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 665
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 668
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2118
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2651
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2700
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 2860
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3364
type: DSZ, layer: 1, pos: 3374
type: DSZ, layer: 1, pos: 3381
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3529
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 375

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0159141, upper bound: 0.0162129
time: 4.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0159202, upper bound: 0.0162078
time: 13.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.8993219, -0.2406590, -0.8993219, -0.2406590, -0.3864202, 0.3866106
1: -0.0854008, 0.3246739, -0.0854008, 0.3246739, -0.0648450, 0.0647171
2: -2.2585669, -1.4722774, -2.2585669, -1.4722774, -0.2641876, 0.2642084
3: -2.8736234, -1.7064936, -2.8736234, -1.7064936, -0.4017511, 0.4017068
4: -3.5778260, -2.4280591, -3.5778260, -2.4280591, -0.4159189, 0.4157206
5: -2.5908875, -1.5438142, -2.5908875, -1.5438142, -0.2096939, 0.2096542
6: -2.5916283, -1.4587758, -2.5916283, -1.4587758, -0.1933999, 0.1932245
7: -2.8178353, -1.4403852, -2.8178353, -1.4403852, -0.5062618, 0.5059665
8: -0.0413489, 0.2033695, -0.0413489, 0.2033695, -0.1476315, 0.1476099
9: -0.8593886, -0.4630676, -0.8593886, -0.4630676, -0.2098528, 0.2098081

Time for backsubstitution: 6.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 3264
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 345
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 377
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 2859
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 376
type: DSZ, layer: 1, pos: 3501
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 433
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 665
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 668
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2118
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2651
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2700
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 2860
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3364
type: DSZ, layer: 1, pos: 3374
type: DSZ, layer: 1, pos: 3381
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3529
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 375

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0162039, upper bound: 0.0159209
time: 142.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0162037, upper bound: 0.0159170
time: 55.96 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 205.06 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 205.06
Output dim: 1, lower bound: -0.0159141, upper bound: 0.0162129
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 205.06
Output dim: 1, lower bound: -0.0159202, upper bound: 0.0162078
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 205.06
Output dim: 1, lower bound: -0.0162039, upper bound: 0.0159209
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 205.06
Output dim: 1, lower bound: -0.0162037, upper bound: 0.0159170

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.8993219, -0.2406590, -0.8993219, -0.2406590, -0.3856328, 0.3855834
1: -0.0854008, 0.3246739, -0.0854008, 0.3246739, -0.0647166, 0.0648470
2: -2.2585669, -1.4722774, -2.2585669, -1.4722774, -0.2642182, 0.2641059
3: -2.8736234, -1.7064936, -2.8736234, -1.7064936, -0.4011152, 0.4010596
4: -3.5778260, -2.4280591, -3.5778260, -2.4280591, -0.4149088, 0.4149721
5: -2.5908875, -1.5438142, -2.5908875, -1.5438142, -0.2092012, 0.2091646
6: -2.5916283, -1.4587758, -2.5916283, -1.4587758, -0.1927505, 0.1928590
7: -2.8178353, -1.4403852, -2.8178353, -1.4403852, -0.5046254, 0.5045407
8: -0.0413489, 0.2033695, -0.0413489, 0.2033695, -0.1476099, 0.1476462
9: -0.8593886, -0.4630676, -0.8593886, -0.4630676, -0.2097982, 0.2098427

Time for backsubstitution: 6.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 3264
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 345
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 377
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 2859
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 376
type: DSZ, layer: 1, pos: 3501
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 433
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 665
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 668
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2118
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2651
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2700
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 2860
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3364
type: DSZ, layer: 1, pos: 3374
type: DSZ, layer: 1, pos: 3381
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3529
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3502

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0159113, upper bound: 0.0162126
time: 4.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0159141, upper bound: 0.0162163
time: 3.93 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.8993219, -0.2406590, -0.8993219, -0.2406590, -0.3857738, 0.3854424
1: -0.0854008, 0.3246739, -0.0854008, 0.3246739, -0.0647202, 0.0648446
2: -2.2585669, -1.4722774, -2.2585669, -1.4722774, -0.2641268, 0.2641972
3: -2.8736234, -1.7064936, -2.8736234, -1.7064936, -0.4010153, 0.4011595
4: -3.5778260, -2.4280591, -3.5778260, -2.4280591, -0.4147736, 0.4151095
5: -2.5908875, -1.5438142, -2.5908875, -1.5438142, -0.2091248, 0.2092409
6: -2.5916283, -1.4587758, -2.5916283, -1.4587758, -0.1926836, 0.1929259
7: -2.8178353, -1.4403852, -2.8178353, -1.4403852, -0.5042455, 0.5049206
8: -0.0413489, 0.2033695, -0.0413489, 0.2033695, -0.1476247, 0.1476315
9: -0.8593886, -0.4630676, -0.8593886, -0.4630676, -0.2097980, 0.2098429

Time for backsubstitution: 6.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 3264
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 345
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 377
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 2859
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 376
type: DSZ, layer: 1, pos: 3501
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 433
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 665
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 668
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2118
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2651
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2700
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 2860
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3364
type: DSZ, layer: 1, pos: 3374
type: DSZ, layer: 1, pos: 3381
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3529
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 3502

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0159174, upper bound: 0.0162135
time: 5.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0159202, upper bound: 0.0161177
time: 51.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.8993219, -0.2406590, -0.8993219, -0.2406590, -0.3854424, 0.3857737
1: -0.0854008, 0.3246739, -0.0854008, 0.3246739, -0.0648446, 0.0647202
2: -2.2585669, -1.4722774, -2.2585669, -1.4722774, -0.2641973, 0.2641268
3: -2.8736234, -1.7064936, -2.8736234, -1.7064936, -0.4011595, 0.4010153
4: -3.5778260, -2.4280591, -3.5778260, -2.4280591, -0.4151094, 0.4147737
5: -2.5908875, -1.5438142, -2.5908875, -1.5438142, -0.2092409, 0.2091248
6: -2.5916283, -1.4587758, -2.5916283, -1.4587758, -0.1929259, 0.1926835
7: -2.8178353, -1.4403852, -2.8178353, -1.4403852, -0.5049206, 0.5042455
8: -0.0413489, 0.2033695, -0.0413489, 0.2033695, -0.1476314, 0.1476247
9: -0.8593886, -0.4630676, -0.8593886, -0.4630676, -0.2098429, 0.2097981

Time for backsubstitution: 6.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 3264
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 345
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 377
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 2859
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 376
type: DSZ, layer: 1, pos: 3501
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 433
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 665
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 668
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2118
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2651
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2700
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 2860
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3364
type: DSZ, layer: 1, pos: 3374
type: DSZ, layer: 1, pos: 3381
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3529
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 3502

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0162010, upper bound: 0.0159305
time: 11.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0162038, upper bound: 0.0159217
time: 12.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.8993219, -0.2406590, -0.8993219, -0.2406590, -0.3855834, 0.3856327
1: -0.0854008, 0.3246739, -0.0854008, 0.3246739, -0.0648470, 0.0647166
2: -2.2585669, -1.4722774, -2.2585669, -1.4722774, -0.2641059, 0.2642182
3: -2.8736234, -1.7064936, -2.8736234, -1.7064936, -0.4010596, 0.4011152
4: -3.5778260, -2.4280591, -3.5778260, -2.4280591, -0.4149720, 0.4149089
5: -2.5908875, -1.5438142, -2.5908875, -1.5438142, -0.2091645, 0.2092012
6: -2.5916283, -1.4587758, -2.5916283, -1.4587758, -0.1928590, 0.1927505
7: -2.8178353, -1.4403852, -2.8178353, -1.4403852, -0.5045407, 0.5046254
8: -0.0413489, 0.2033695, -0.0413489, 0.2033695, -0.1476462, 0.1476099
9: -0.8593886, -0.4630676, -0.8593886, -0.4630676, -0.2098427, 0.2097982

Time for backsubstitution: 6.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 3264
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 345
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 377
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 2859
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 376
type: DSZ, layer: 1, pos: 3501
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 433
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 665
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 668
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2118
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2651
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2700
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 2860
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3364
type: DSZ, layer: 1, pos: 3374
type: DSZ, layer: 1, pos: 3381
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3529
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 3502

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0162009, upper bound: 0.0159210
time: 9.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0162037, upper bound: 0.0159179
time: 56.86 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 73.01 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 73.01
Output dim: 1, lower bound: -0.0159113, upper bound: 0.0162126
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 73.01
Output dim: 1, lower bound: -0.0159141, upper bound: 0.0162163
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 73.01
Output dim: 1, lower bound: -0.0159174, upper bound: 0.0162135
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 73.01
Output dim: 1, lower bound: -0.0159202, upper bound: 0.0161177
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 73.01
Output dim: 1, lower bound: -0.0162010, upper bound: 0.0159305
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 73.01
Output dim: 1, lower bound: -0.0162038, upper bound: 0.0159217
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 73.01
Output dim: 1, lower bound: -0.0162009, upper bound: 0.0159210
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 73.01
Output dim: 1, lower bound: -0.0162037, upper bound: 0.0159179

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.8993219, -0.2406590, -0.8993219, -0.2406590, -0.3852530, 0.3851967
1: -0.0854008, 0.3246739, -0.0854008, 0.3246739, -0.0643260, 0.0644589
2: -2.2585669, -1.4722774, -2.2585669, -1.4722774, -0.2610929, 0.2608656
3: -2.8736234, -1.7064936, -2.8736234, -1.7064936, -0.3993523, 0.3992796
4: -3.5778260, -2.4280591, -3.5778260, -2.4280591, -0.4126488, 0.4126641
5: -2.5908875, -1.5438142, -2.5908875, -1.5438142, -0.2067634, 0.2067114
6: -2.5916283, -1.4587758, -2.5916283, -1.4587758, -0.1908073, 0.1909524
7: -2.8178353, -1.4403852, -2.8178353, -1.4403852, -0.4998651, 0.4996649
8: -0.0413489, 0.2033695, -0.0413489, 0.2033695, -0.1473573, 0.1474043
9: -0.8593886, -0.4630676, -0.8593886, -0.4630676, -0.2086183, 0.2086149

Time for backsubstitution: 6.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 3264
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 345
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 377
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 2859
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 376
type: DSZ, layer: 1, pos: 3501
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 433
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 665
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 668
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2118
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2651
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2700
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 2860
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3364
type: DSZ, layer: 1, pos: 3374
type: DSZ, layer: 1, pos: 3381
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3529
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 2850

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0159057, upper bound: 0.0159804
time: 121.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0159056, upper bound: 0.0162089
time: 5.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.8993219, -0.2406590, -0.8993219, -0.2406590, -0.3852460, 0.3852036
1: -0.0854008, 0.3246739, -0.0854008, 0.3246739, -0.0643285, 0.0644565
2: -2.2585669, -1.4722774, -2.2585669, -1.4722774, -0.2609779, 0.2609806
3: -2.8736234, -1.7064936, -2.8736234, -1.7064936, -0.3993352, 0.3992967
4: -3.5778260, -2.4280591, -3.5778260, -2.4280591, -0.4126010, 0.4127120
5: -2.5908875, -1.5438142, -2.5908875, -1.5438142, -0.2067480, 0.2067267
6: -2.5916283, -1.4587758, -2.5916283, -1.4587758, -0.1908439, 0.1909158
7: -2.8178353, -1.4403852, -2.8178353, -1.4403852, -0.4997497, 0.4997803
8: -0.0413489, 0.2033695, -0.0413489, 0.2033695, -0.1473680, 0.1473936
9: -0.8593886, -0.4630676, -0.8593886, -0.4630676, -0.2085704, 0.2086628

Time for backsubstitution: 6.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 3264
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 345
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 377
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 2859
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 376
type: DSZ, layer: 1, pos: 3501
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 433
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 665
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 668
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2118
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2651
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2700
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 2860
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3364
type: DSZ, layer: 1, pos: 3374
type: DSZ, layer: 1, pos: 3381
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3529
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2850

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0159085, upper bound: 0.0162033
time: 4.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0159084, upper bound: 0.0161931
time: 49.53 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.8993219, -0.2406590, -0.8993219, -0.2406590, -0.3853940, 0.3850557
1: -0.0854008, 0.3246739, -0.0854008, 0.3246739, -0.0643296, 0.0644565
2: -2.2585669, -1.4722774, -2.2585669, -1.4722774, -0.2610015, 0.2609570
3: -2.8736234, -1.7064936, -2.8736234, -1.7064936, -0.3992524, 0.3993795
4: -3.5778260, -2.4280591, -3.5778260, -2.4280591, -0.4125135, 0.4128015
5: -2.5908875, -1.5438142, -2.5908875, -1.5438142, -0.2066870, 0.2067877
6: -2.5916283, -1.4587758, -2.5916283, -1.4587758, -0.1907404, 0.1910194
7: -2.8178353, -1.4403852, -2.8178353, -1.4403852, -0.4994851, 0.5000448
8: -0.0413489, 0.2033695, -0.0413489, 0.2033695, -0.1473721, 0.1473895
9: -0.8593886, -0.4630676, -0.8593886, -0.4630676, -0.2086182, 0.2086150

Time for backsubstitution: 6.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 3264
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 345
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 377
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 2859
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 376
type: DSZ, layer: 1, pos: 3501
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 433
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 665
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 668
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2118
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2651
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2700
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 2860
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3364
type: DSZ, layer: 1, pos: 3374
type: DSZ, layer: 1, pos: 3381
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3529
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2850

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0159102, upper bound: 0.0161995
time: 5.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0159095, upper bound: 0.0162078
time: 4.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.8993219, -0.2406590, -0.8993219, -0.2406590, -0.3850626, 0.3853870
1: -0.0854008, 0.3246739, -0.0854008, 0.3246739, -0.0644540, 0.0643321
2: -2.2585669, -1.4722774, -2.2585669, -1.4722774, -0.2610720, 0.2608865
3: -2.8736234, -1.7064936, -2.8736234, -1.7064936, -0.3993966, 0.3992353
4: -3.5778260, -2.4280591, -3.5778260, -2.4280591, -0.4128493, 0.4124658
5: -2.5908875, -1.5438142, -2.5908875, -1.5438142, -0.2068031, 0.2066717
6: -2.5916283, -1.4587758, -2.5916283, -1.4587758, -0.1909828, 0.1907770
7: -2.8178353, -1.4403852, -2.8178353, -1.4403852, -0.5001603, 0.4993697
8: -0.0413489, 0.2033695, -0.0413489, 0.2033695, -0.1473788, 0.1473828
9: -0.8593886, -0.4630676, -0.8593886, -0.4630676, -0.2086630, 0.2085703

Time for backsubstitution: 6.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 3264
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 345
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 377
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 2859
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 376
type: DSZ, layer: 1, pos: 3501
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 433
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 665
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 668
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2118
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2651
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2700
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 2860
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3364
type: DSZ, layer: 1, pos: 3374
type: DSZ, layer: 1, pos: 3381
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3529
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2850

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0161941, upper bound: 0.0159145
time: 103.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0161934, upper bound: 0.0159258
time: 4.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.8993219, -0.2406590, -0.8993219, -0.2406590, -0.3850556, 0.3853940
1: -0.0854008, 0.3246739, -0.0854008, 0.3246739, -0.0644565, 0.0643296
2: -2.2585669, -1.4722774, -2.2585669, -1.4722774, -0.2609570, 0.2610015
3: -2.8736234, -1.7064936, -2.8736234, -1.7064936, -0.3993795, 0.3992524
4: -3.5778260, -2.4280591, -3.5778260, -2.4280591, -0.4128015, 0.4125136
5: -2.5908875, -1.5438142, -2.5908875, -1.5438142, -0.2067877, 0.2066870
6: -2.5916283, -1.4587758, -2.5916283, -1.4587758, -0.1910194, 0.1907404
7: -2.8178353, -1.4403852, -2.8178353, -1.4403852, -0.5000449, 0.4994851
8: -0.0413489, 0.2033695, -0.0413489, 0.2033695, -0.1473895, 0.1473721
9: -0.8593886, -0.4630676, -0.8593886, -0.4630676, -0.2086150, 0.2086182

Time for backsubstitution: 6.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 3264
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 345
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 377
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 2859
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 376
type: DSZ, layer: 1, pos: 3501
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 433
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 665
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 668
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2118
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2651
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2700
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 2860
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3364
type: DSZ, layer: 1, pos: 3374
type: DSZ, layer: 1, pos: 3381
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3529
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2850

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0161968, upper bound: 0.0159227
time: 6.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0161962, upper bound: 0.0159110
time: 16.71 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.8993219, -0.2406590, -0.8993219, -0.2406590, -0.3852037, 0.3852460
1: -0.0854008, 0.3246739, -0.0854008, 0.3246739, -0.0644565, 0.0643285
2: -2.2585669, -1.4722774, -2.2585669, -1.4722774, -0.2609807, 0.2609779
3: -2.8736234, -1.7064936, -2.8736234, -1.7064936, -0.3992966, 0.3993352
4: -3.5778260, -2.4280591, -3.5778260, -2.4280591, -0.4127119, 0.4126010
5: -2.5908875, -1.5438142, -2.5908875, -1.5438142, -0.2067267, 0.2067480
6: -2.5916283, -1.4587758, -2.5916283, -1.4587758, -0.1909158, 0.1908439
7: -2.8178353, -1.4403852, -2.8178353, -1.4403852, -0.4997804, 0.4997497
8: -0.0413489, 0.2033695, -0.0413489, 0.2033695, -0.1473936, 0.1473680
9: -0.8593886, -0.4630676, -0.8593886, -0.4630676, -0.2086628, 0.2085704

Time for backsubstitution: 6.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 3264
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 345
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 377
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 2859
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 376
type: DSZ, layer: 1, pos: 3501
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 433
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 665
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 668
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2118
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2651
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2700
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 2860
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3364
type: DSZ, layer: 1, pos: 3374
type: DSZ, layer: 1, pos: 3381
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3529
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2850

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0161938, upper bound: 0.0159163
time: 84.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0161930, upper bound: 0.0159124
time: 24.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.8993219, -0.2406590, -0.8993219, -0.2406590, -0.3851966, 0.3852530
1: -0.0854008, 0.3246739, -0.0854008, 0.3246739, -0.0644589, 0.0643260
2: -2.2585669, -1.4722774, -2.2585669, -1.4722774, -0.2608657, 0.2610929
3: -2.8736234, -1.7064936, -2.8736234, -1.7064936, -0.3992795, 0.3993524
4: -3.5778260, -2.4280591, -3.5778260, -2.4280591, -0.4126640, 0.4126489
5: -2.5908875, -1.5438142, -2.5908875, -1.5438142, -0.2067114, 0.2067634
6: -2.5916283, -1.4587758, -2.5916283, -1.4587758, -0.1909524, 0.1908073
7: -2.8178353, -1.4403852, -2.8178353, -1.4403852, -0.4996650, 0.4998651
8: -0.0413489, 0.2033695, -0.0413489, 0.2033695, -0.1474043, 0.1473573
9: -0.8593886, -0.4630676, -0.8593886, -0.4630676, -0.2086149, 0.2086183

Time for backsubstitution: 6.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 3264
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 345
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 377
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 2859
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 376
type: DSZ, layer: 1, pos: 3501
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 433
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 665
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 668
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2118
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2651
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2700
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 2860
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3364
type: DSZ, layer: 1, pos: 3374
type: DSZ, layer: 1, pos: 3381
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3529
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2850

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0161966, upper bound: 0.0159136
time: 4.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0161958, upper bound: 0.0159119
time: 52.64 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 63.14 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 63.14
Output dim: 1, lower bound: -0.0159057, upper bound: 0.0159804
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 63.14
Output dim: 1, lower bound: -0.0159056, upper bound: 0.0162089
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 63.14
Output dim: 1, lower bound: -0.0159085, upper bound: 0.0162033
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 63.14
Output dim: 1, lower bound: -0.0159084, upper bound: 0.0161931
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 63.14
Output dim: 1, lower bound: -0.0159102, upper bound: 0.0161995
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 63.14
Output dim: 1, lower bound: -0.0159095, upper bound: 0.0162078
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 63.14
Output dim: 1, lower bound: -0.0161941, upper bound: 0.0159145
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 63.14
Output dim: 1, lower bound: -0.0161934, upper bound: 0.0159258
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 63.14
Output dim: 1, lower bound: -0.0161968, upper bound: 0.0159227
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 63.14
Output dim: 1, lower bound: -0.0161962, upper bound: 0.0159110
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 63.14
Output dim: 1, lower bound: -0.0161938, upper bound: 0.0159163
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 63.14
Output dim: 1, lower bound: -0.0161930, upper bound: 0.0159124
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 63.14
Output dim: 1, lower bound: -0.0161966, upper bound: 0.0159136
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 63.14
Output dim: 1, lower bound: -0.0161958, upper bound: 0.0159119

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.8993219, -0.2406590, -0.8993219, -0.2406590, -0.3852527, 0.3851965
1: -0.0854008, 0.3246739, -0.0854008, 0.3246739, -0.0641695, 0.0643069
2: -2.2585669, -1.4722774, -2.2585669, -1.4722774, -0.2610769, 0.2608497
3: -2.8736234, -1.7064936, -2.8736234, -1.7064936, -0.3993507, 0.3992779
4: -3.5778260, -2.4280591, -3.5778260, -2.4280591, -0.4126221, 0.4126375
5: -2.5908875, -1.5438142, -2.5908875, -1.5438142, -0.2067633, 0.2067113
6: -2.5916283, -1.4587758, -2.5916283, -1.4587758, -0.1908042, 0.1909494
7: -2.8178353, -1.4403852, -2.8178353, -1.4403852, -0.4997835, 0.4995834
8: -0.0413489, 0.2033695, -0.0413489, 0.2033695, -0.1473536, 0.1474011
9: -0.8593886, -0.4630676, -0.8593886, -0.4630676, -0.2085388, 0.2085378

Time for backsubstitution: 6.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 3264
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 345
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 377
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 2859
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 376
type: DSZ, layer: 1, pos: 3501
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 433
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 665
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 668
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2118
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2651
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2700
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 2860
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3364
type: DSZ, layer: 1, pos: 3374
type: DSZ, layer: 1, pos: 3381
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3529
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2836

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0159018, upper bound: 0.0162104
time: 3.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0159023, upper bound: 0.0162004
time: 63.35 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.8993219, -0.2406590, -0.8993219, -0.2406590, -0.3852458, 0.3852035
1: -0.0854008, 0.3246739, -0.0854008, 0.3246739, -0.0641765, 0.0642985
2: -2.2585669, -1.4722774, -2.2585669, -1.4722774, -0.2609619, 0.2609646
3: -2.8736234, -1.7064936, -2.8736234, -1.7064936, -0.3993336, 0.3992950
4: -3.5778260, -2.4280591, -3.5778260, -2.4280591, -0.4125744, 0.4126852
5: -2.5908875, -1.5438142, -2.5908875, -1.5438142, -0.2067479, 0.2067266
6: -2.5916283, -1.4587758, -2.5916283, -1.4587758, -0.1908409, 0.1909118
7: -2.8178353, -1.4403852, -2.8178353, -1.4403852, -0.4996682, 0.4996988
8: -0.0413489, 0.2033695, -0.0413489, 0.2033695, -0.1473649, 0.1473899
9: -0.8593886, -0.4630676, -0.8593886, -0.4630676, -0.2084933, 0.2085833

Time for backsubstitution: 6.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2836
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 3264
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 345
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 377
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 2859
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 376
type: DSZ, layer: 1, pos: 3501
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 433
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 665
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 668
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 670
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2118
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2651
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2700
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2710
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 2860
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3108
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3363
type: DSZ, layer: 1, pos: 3364
type: DSZ, layer: 1, pos: 3374
type: DSZ, layer: 1, pos: 3381
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3529
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2836

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0159046, upper bound: 0.0162012
time: 4.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0159052, upper bound: 0.0161897
time: 153.30 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 164.31 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 164.31
Output dim: 1, lower bound: -0.0159018, upper bound: 0.0162104
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 164.31
Output dim: 1, lower bound: -0.0159023, upper bound: 0.0162004
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 164.31
Output dim: 1, lower bound: -0.0159046, upper bound: 0.0162012
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 164.31
Output dim: 1, lower bound: -0.0159052, upper bound: 0.0161897
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 164.31
Output dim: 1, lower bound: -0.0159084, upper bound: 0.0161931
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 164.31
Output dim: 1, lower bound: -0.0159102, upper bound: 0.0161995
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 164.31
Output dim: 1, lower bound: -0.0159095, upper bound: 0.0162078
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 164.31
Output dim: 1, lower bound: -0.0161941, upper bound: 0.0159145
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 164.31
Output dim: 1, lower bound: -0.0161934, upper bound: 0.0159258
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 164.31
Output dim: 1, lower bound: -0.0161968, upper bound: 0.0159227
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 164.31
Output dim: 1, lower bound: -0.0161962, upper bound: 0.0159110
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 164.31
Output dim: 1, lower bound: -0.0161938, upper bound: 0.0159163
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 164.31
Output dim: 1, lower bound: -0.0161930, upper bound: 0.0159124
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 164.31
Output dim: 1, lower bound: -0.0161966, upper bound: 0.0159136
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 164.31
Output dim: 1, lower bound: -0.0161958, upper bound: 0.0159119

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 36.12 + 1885.81 = 1921.93 seconds
