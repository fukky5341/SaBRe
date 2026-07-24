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
execution time: IAR + RelationalAnalysis = 7.87 + 28.31 = 36.17 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0162071, upper bound: 0.0162100

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3106
type: A, layer: 1, pos: 2965
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2966
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2100
type: A, layer: 1, pos: 3108
type: A, layer: 1, pos: 2431
type: A, layer: 1, pos: 2101
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 2122
type: A, layer: 1, pos: 2333
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 345
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 3022
type: A, layer: 1, pos: 390
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 2703
type: A, layer: 1, pos: 2995
type: A, layer: 1, pos: 2710
type: A, layer: 1, pos: 2701
type: A, layer: 1, pos: 2702
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 2700
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 380
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3421
type: A, layer: 1, pos: 2132
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 2120
type: A, layer: 1, pos: 2336
type: A, layer: 1, pos: 2865
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 2097
type: A, layer: 1, pos: 2704
type: A, layer: 1, pos: 2705
type: A, layer: 1, pos: 3538
type: A, layer: 1, pos: 2850
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 2096
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 377
type: A, layer: 1, pos: 379
type: A, layer: 1, pos: 2133
type: A, layer: 1, pos: 2321
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 3259
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 376
type: A, layer: 1, pos: 2225
type: A, layer: 1, pos: 2820
type: A, layer: 1, pos: 3264
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 2862
type: A, layer: 1, pos: 2435
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 2994
type: A, layer: 1, pos: 2706
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2118
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2102
type: A, layer: 1, pos: 378
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 2709
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 274
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 3260
type: A, layer: 1, pos: 2860
type: A, layer: 1, pos: 3266
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 2891
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 2836
type: A, layer: 1, pos: 2855
type: A, layer: 1, pos: 2840
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 272
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 2116
type: A, layer: 1, pos: 3128
type: A, layer: 1, pos: 2651
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 2846
type: A, layer: 1, pos: 2858
type: A, layer: 1, pos: 3381
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3544
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 3529
type: A, layer: 1, pos: 2649
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2859
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 2413
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 433
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 3501
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 375
type: A, layer: 1, pos: 3383
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 3502
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 3343
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 509
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2099
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2129
type: A, layer: 1, pos: 2264
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2697
type: A, layer: 1, pos: 2864
type: A, layer: 1, pos: 3361
type: A, layer: 1, pos: 3363
type: A, layer: 1, pos: 3364
type: A, layer: 1, pos: 3374
type: A, layer: 1, pos: 3554
type: A, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0160808, upper bound: 0.0162073
time: 85.08 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0162048, upper bound: 0.0162189
time: 5.44 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 90.59 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 90.59
Output dim: 1, lower bound: -0.0160808, upper bound: 0.0162073
NS_A2, status: Status.UNKNOWN, split count: 1, time: 90.59
Output dim: 1, lower bound: -0.0162048, upper bound: 0.0162189

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.8988743, -0.2409335, -0.8992715, -0.2408981, -0.3920150, 0.3923587
1: -0.0834007, 0.3234223, -0.0836493, 0.3246737, -0.0648048, 0.0637245
2: -2.2584682, -1.4724362, -2.2585604, -1.4724150, -0.2648688, 0.2648534
3: -2.8732057, -1.7067522, -2.8732543, -1.7065008, -0.4073289, 0.4071023
4: -3.5774922, -2.4287608, -3.5777979, -2.4287021, -0.4181054, 0.4184730
5: -2.5904577, -1.5440657, -2.5905054, -1.5438209, -0.2143391, 0.2141190
6: -2.5907776, -1.4593184, -2.5908883, -1.4587767, -0.1976759, 0.1972463
7: -2.8177798, -1.4404993, -2.8178174, -1.4404861, -0.5124356, 0.5124621
8: -0.0411193, 0.2033566, -0.0413320, 0.2033583, -0.1473865, 0.1475937
9: -0.8586028, -0.4635347, -0.8586821, -0.4630713, -0.2104070, 0.2099749

Time for backsubstitution: 6.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3106
type: B, layer: 1, pos: 2965
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2966
type: B, layer: 1, pos: 3492
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2100
type: B, layer: 1, pos: 3108
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 2101
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2122
type: B, layer: 1, pos: 2333
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 345
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 3022
type: B, layer: 1, pos: 390
type: B, layer: 1, pos: 2221
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 2586
type: B, layer: 1, pos: 2703
type: B, layer: 1, pos: 2995
type: B, layer: 1, pos: 2710
type: B, layer: 1, pos: 2701
type: B, layer: 1, pos: 2702
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 2700
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 380
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3421
type: B, layer: 1, pos: 2132
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 2120
type: B, layer: 1, pos: 2336
type: B, layer: 1, pos: 2865
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 2617
type: B, layer: 1, pos: 2097
type: B, layer: 1, pos: 2704
type: B, layer: 1, pos: 2705
type: B, layer: 1, pos: 3538
type: B, layer: 1, pos: 2850
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 2096
type: B, layer: 1, pos: 2375
type: B, layer: 1, pos: 377
type: B, layer: 1, pos: 379
type: B, layer: 1, pos: 2133
type: B, layer: 1, pos: 2321
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 3259
type: B, layer: 1, pos: 2600
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 376
type: B, layer: 1, pos: 2225
type: B, layer: 1, pos: 2820
type: B, layer: 1, pos: 3264
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 2862
type: B, layer: 1, pos: 2435
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 2994
type: B, layer: 1, pos: 2706
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2118
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2102
type: B, layer: 1, pos: 378
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2117
type: B, layer: 1, pos: 2709
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 274
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 3260
type: B, layer: 1, pos: 2860
type: B, layer: 1, pos: 3266
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 2891
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 2836
type: B, layer: 1, pos: 2855
type: B, layer: 1, pos: 2840
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 272
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 2116
type: B, layer: 1, pos: 3128
type: B, layer: 1, pos: 2651
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 2846
type: B, layer: 1, pos: 2858
type: B, layer: 1, pos: 3381
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3544
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 3529
type: B, layer: 1, pos: 2649
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 2859
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2666
type: B, layer: 1, pos: 433
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2232
type: B, layer: 1, pos: 3501
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 375
type: B, layer: 1, pos: 3383
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 3502
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 3343
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 509
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 2069
type: B, layer: 1, pos: 2099
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2129
type: B, layer: 1, pos: 2264
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2697
type: B, layer: 1, pos: 2864
type: B, layer: 1, pos: 3361
type: B, layer: 1, pos: 3363
type: B, layer: 1, pos: 3364
type: B, layer: 1, pos: 3374
type: B, layer: 1, pos: 3554
type: B, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3106

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0160796, upper bound: 0.0161161
time: 3.49 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0160795, upper bound: 0.0162037
time: 110.53 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.8993197, -0.2406622, -0.8993201, -0.2406618, -0.3926637, 0.3925446
1: -0.0853678, 0.3246738, -0.0853722, 0.3246738, -0.0635678, 0.0668675
2: -2.2585645, -1.4722791, -2.2585647, -1.4722791, -0.2649070, 0.2648856
3: -2.8736148, -1.7064952, -2.8736155, -1.7064948, -0.4070908, 0.4077364
4: -3.5778246, -2.4280722, -3.5778251, -2.4280705, -0.4192493, 0.4182339
5: -2.5908809, -1.5438139, -2.5908821, -1.5438139, -0.2141058, 0.2147753
6: -2.5916181, -1.4587762, -2.5916195, -1.4587758, -0.1971860, 0.1985173
7: -2.8178329, -1.4403875, -2.8178334, -1.4403874, -0.5125867, 0.5125157
8: -0.0413483, 0.2033598, -0.0413484, 0.2033611, -0.1476140, 0.1476142
9: -0.8593743, -0.4630679, -0.8593761, -0.4630678, -0.2099220, 0.2112223

Time for backsubstitution: 6.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3106
type: B, layer: 1, pos: 2965
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2966
type: B, layer: 1, pos: 3492
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2100
type: B, layer: 1, pos: 3108
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 2101
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2122
type: B, layer: 1, pos: 2333
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 345
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 3022
type: B, layer: 1, pos: 390
type: B, layer: 1, pos: 2221
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 2586
type: B, layer: 1, pos: 2703
type: B, layer: 1, pos: 2995
type: B, layer: 1, pos: 2710
type: B, layer: 1, pos: 2701
type: B, layer: 1, pos: 2702
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 2700
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 380
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3421
type: B, layer: 1, pos: 2132
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 2120
type: B, layer: 1, pos: 2336
type: B, layer: 1, pos: 2865
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 2617
type: B, layer: 1, pos: 2097
type: B, layer: 1, pos: 2704
type: B, layer: 1, pos: 2705
type: B, layer: 1, pos: 3538
type: B, layer: 1, pos: 2850
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 2096
type: B, layer: 1, pos: 2375
type: B, layer: 1, pos: 377
type: B, layer: 1, pos: 379
type: B, layer: 1, pos: 2133
type: B, layer: 1, pos: 2321
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 3259
type: B, layer: 1, pos: 2600
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 376
type: B, layer: 1, pos: 2225
type: B, layer: 1, pos: 2820
type: B, layer: 1, pos: 3264
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 2862
type: B, layer: 1, pos: 2435
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 2994
type: B, layer: 1, pos: 2706
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2118
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2102
type: B, layer: 1, pos: 378
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2117
type: B, layer: 1, pos: 2709
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 274
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 3260
type: B, layer: 1, pos: 2860
type: B, layer: 1, pos: 3266
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 2891
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 2836
type: B, layer: 1, pos: 2855
type: B, layer: 1, pos: 2840
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 272
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 2116
type: B, layer: 1, pos: 3128
type: B, layer: 1, pos: 2651
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 2846
type: B, layer: 1, pos: 2858
type: B, layer: 1, pos: 3381
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3544
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 3529
type: B, layer: 1, pos: 2649
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 2859
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2666
type: B, layer: 1, pos: 433
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2232
type: B, layer: 1, pos: 3501
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 375
type: B, layer: 1, pos: 3383
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 3502
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 3343
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 509
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 2069
type: B, layer: 1, pos: 2099
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2129
type: B, layer: 1, pos: 2264
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2697
type: B, layer: 1, pos: 2864
type: B, layer: 1, pos: 3361
type: B, layer: 1, pos: 3363
type: B, layer: 1, pos: 3364
type: B, layer: 1, pos: 3374
type: B, layer: 1, pos: 3554
type: B, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3106

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0162031, upper bound: 0.0161003
time: 79.20 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0162030, upper bound: 0.0162026
time: 135.49 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 221.13 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 221.13
Output dim: 1, lower bound: -0.0160796, upper bound: 0.0161161
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 221.13
Output dim: 1, lower bound: -0.0160795, upper bound: 0.0162037
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 221.13
Output dim: 1, lower bound: -0.0162031, upper bound: 0.0161003
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 221.13
Output dim: 1, lower bound: -0.0162030, upper bound: 0.0162026

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.8988686, -0.2409349, -0.8992649, -0.2409000, -0.3920071, 0.3923339
1: -0.0833853, 0.3234221, -0.0836316, 0.3246734, -0.0647815, 0.0610001
2: -2.2584455, -1.4724365, -2.2585344, -1.4724157, -0.2648224, 0.2646045
3: -2.8731990, -1.7068624, -2.8732460, -1.7066275, -0.4072761, 0.4070734
4: -3.5774915, -2.4287868, -3.5777969, -2.4287326, -0.4181186, 0.4183375
5: -2.5904574, -1.5440779, -2.5905044, -1.5438352, -0.2143229, 0.2141165
6: -2.5907633, -1.4593198, -2.5908711, -1.4587781, -0.1976725, 0.1971729
7: -2.8177140, -1.4404993, -2.8177490, -1.4404861, -0.5123787, 0.5116206
8: -0.0411184, 0.2033551, -0.0413310, 0.2033565, -0.1471900, 0.1475782
9: -0.8585816, -0.4635357, -0.8586580, -0.4630724, -0.2102019, 0.2084481

Time for backsubstitution: 6.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2965
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2966
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2100
type: A, layer: 1, pos: 3108
type: A, layer: 1, pos: 2431
type: A, layer: 1, pos: 2101
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 2122
type: A, layer: 1, pos: 2333
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 345
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 3022
type: A, layer: 1, pos: 390
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 2703
type: A, layer: 1, pos: 2995
type: A, layer: 1, pos: 2710
type: A, layer: 1, pos: 2701
type: A, layer: 1, pos: 2702
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 2700
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 380
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3421
type: A, layer: 1, pos: 3106
type: A, layer: 1, pos: 2132
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 2120
type: A, layer: 1, pos: 2336
type: A, layer: 1, pos: 2865
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 2097
type: A, layer: 1, pos: 2704
type: A, layer: 1, pos: 2705
type: A, layer: 1, pos: 3538
type: A, layer: 1, pos: 2850
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 2096
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 377
type: A, layer: 1, pos: 379
type: A, layer: 1, pos: 2133
type: A, layer: 1, pos: 2321
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 3259
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 376
type: A, layer: 1, pos: 2225
type: A, layer: 1, pos: 2820
type: A, layer: 1, pos: 3264
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 2862
type: A, layer: 1, pos: 2435
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 2994
type: A, layer: 1, pos: 2706
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2118
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2102
type: A, layer: 1, pos: 378
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 2709
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 274
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 3260
type: A, layer: 1, pos: 2860
type: A, layer: 1, pos: 3266
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 2891
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 2836
type: A, layer: 1, pos: 2855
type: A, layer: 1, pos: 2840
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 272
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 2116
type: A, layer: 1, pos: 3128
type: A, layer: 1, pos: 2651
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 2846
type: A, layer: 1, pos: 2858
type: A, layer: 1, pos: 3381
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3544
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 3529
type: A, layer: 1, pos: 2649
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2859
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 2413
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 433
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 3501
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 375
type: A, layer: 1, pos: 3383
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 3502
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 3343
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 509
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2099
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2129
type: A, layer: 1, pos: 2264
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2697
type: A, layer: 1, pos: 2864
type: A, layer: 1, pos: 3361
type: A, layer: 1, pos: 3363
type: A, layer: 1, pos: 3364
type: A, layer: 1, pos: 3374
type: A, layer: 1, pos: 3554
type: A, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2965

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0160045, upper bound: 0.0161859
time: 73.94 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0160569, upper bound: 0.0161900
time: 3.44 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.8992992, -0.2406757, -0.8989867, -0.2406930, -0.3925832, 0.3922095
1: -0.0840333, 0.3246735, -0.0838805, 0.3234798, -0.0609363, 0.0653080
2: -2.2584035, -1.4722805, -2.2583797, -1.4723947, -0.2646553, 0.2647169
3: -2.8735495, -1.7067107, -2.8735309, -1.7067815, -0.4066564, 0.4071063
4: -3.5778203, -2.4281187, -3.5778100, -2.4281330, -0.4190912, 0.4181515
5: -2.5908773, -1.5438197, -2.5908880, -1.5438235, -0.2140808, 0.2147414
6: -2.5915682, -1.4587829, -2.5915620, -1.4588106, -0.1970759, 0.1984525
7: -2.8171525, -1.4403889, -2.8169985, -1.4410170, -0.5112876, 0.5116587
8: -0.0413348, 0.2032329, -0.0408549, 0.2032188, -0.1474791, 0.1470443
9: -0.8582623, -0.4630700, -0.8581069, -0.4639561, -0.2082446, 0.2101345

Time for backsubstitution: 6.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2965
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2966
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2100
type: A, layer: 1, pos: 3108
type: A, layer: 1, pos: 2431
type: A, layer: 1, pos: 2101
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 2122
type: A, layer: 1, pos: 2333
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 345
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 3022
type: A, layer: 1, pos: 390
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 2703
type: A, layer: 1, pos: 2995
type: A, layer: 1, pos: 2710
type: A, layer: 1, pos: 2701
type: A, layer: 1, pos: 2702
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 2700
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 380
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3421
type: A, layer: 1, pos: 3106
type: A, layer: 1, pos: 2132
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 2120
type: A, layer: 1, pos: 2336
type: A, layer: 1, pos: 2865
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 2097
type: A, layer: 1, pos: 2704
type: A, layer: 1, pos: 2705
type: A, layer: 1, pos: 3538
type: A, layer: 1, pos: 2850
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 2096
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 377
type: A, layer: 1, pos: 379
type: A, layer: 1, pos: 2133
type: A, layer: 1, pos: 2321
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 3259
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 376
type: A, layer: 1, pos: 2225
type: A, layer: 1, pos: 2820
type: A, layer: 1, pos: 3264
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 2862
type: A, layer: 1, pos: 2435
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 2994
type: A, layer: 1, pos: 2706
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2118
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2102
type: A, layer: 1, pos: 378
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 2709
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 274
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 3260
type: A, layer: 1, pos: 2860
type: A, layer: 1, pos: 3266
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 2891
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 2836
type: A, layer: 1, pos: 2855
type: A, layer: 1, pos: 2840
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 272
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 2116
type: A, layer: 1, pos: 3128
type: A, layer: 1, pos: 2651
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 2846
type: A, layer: 1, pos: 2858
type: A, layer: 1, pos: 3381
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3544
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 3529
type: A, layer: 1, pos: 2649
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2859
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 2413
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 433
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 3501
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 375
type: A, layer: 1, pos: 3383
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 3502
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 3343
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 509
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2099
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2129
type: A, layer: 1, pos: 2264
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2697
type: A, layer: 1, pos: 2864
type: A, layer: 1, pos: 3361
type: A, layer: 1, pos: 3363
type: A, layer: 1, pos: 3364
type: A, layer: 1, pos: 3374
type: A, layer: 1, pos: 3554
type: A, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2965

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0161279, upper bound: 0.0160815
time: 108.63 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0161801, upper bound: 0.0160779
time: 101.08 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.8993139, -0.2406638, -0.8993134, -0.2406636, -0.3926559, 0.3925197
1: -0.0853524, 0.3246736, -0.0853546, 0.3246735, -0.0635445, 0.0641432
2: -2.2585411, -1.4722798, -2.2585387, -1.4722795, -0.2648604, 0.2646368
3: -2.8736062, -1.7066045, -2.8736067, -1.7066207, -0.4070379, 0.4077078
4: -3.5778244, -2.4280987, -3.5778244, -2.4281013, -0.4192623, 0.4180985
5: -2.5908802, -1.5438268, -2.5908802, -1.5438287, -0.2140896, 0.2147727
6: -2.5916040, -1.4587770, -2.5916038, -1.4587772, -0.1971826, 0.1984439
7: -2.8177679, -1.4403880, -2.8177652, -1.4403875, -0.5125300, 0.5116743
8: -0.0413474, 0.2033582, -0.0413473, 0.2033593, -0.1474171, 0.1475986
9: -0.8593532, -0.4630688, -0.8593520, -0.4630689, -0.2097170, 0.2096954

Time for backsubstitution: 6.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2965
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2966
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2100
type: A, layer: 1, pos: 3108
type: A, layer: 1, pos: 2431
type: A, layer: 1, pos: 2101
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 2122
type: A, layer: 1, pos: 2333
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 345
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 3022
type: A, layer: 1, pos: 390
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 2703
type: A, layer: 1, pos: 2995
type: A, layer: 1, pos: 2710
type: A, layer: 1, pos: 2701
type: A, layer: 1, pos: 2702
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 2700
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 380
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3421
type: A, layer: 1, pos: 3106
type: A, layer: 1, pos: 2132
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 2120
type: A, layer: 1, pos: 2336
type: A, layer: 1, pos: 2865
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 2097
type: A, layer: 1, pos: 2704
type: A, layer: 1, pos: 2705
type: A, layer: 1, pos: 3538
type: A, layer: 1, pos: 2850
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 2096
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 377
type: A, layer: 1, pos: 379
type: A, layer: 1, pos: 2133
type: A, layer: 1, pos: 2321
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 3259
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 376
type: A, layer: 1, pos: 2225
type: A, layer: 1, pos: 2820
type: A, layer: 1, pos: 3264
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 2862
type: A, layer: 1, pos: 2435
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 2994
type: A, layer: 1, pos: 2706
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2118
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2102
type: A, layer: 1, pos: 378
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 2709
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 274
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 3260
type: A, layer: 1, pos: 2860
type: A, layer: 1, pos: 3266
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 2891
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 2836
type: A, layer: 1, pos: 2855
type: A, layer: 1, pos: 2840
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 272
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 2116
type: A, layer: 1, pos: 3128
type: A, layer: 1, pos: 2651
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 2846
type: A, layer: 1, pos: 2858
type: A, layer: 1, pos: 3381
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3544
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 3529
type: A, layer: 1, pos: 2649
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2859
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 2413
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 433
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 3501
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 375
type: A, layer: 1, pos: 3383
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 3502
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 3343
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 509
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2099
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2129
type: A, layer: 1, pos: 2264
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2697
type: A, layer: 1, pos: 2864
type: A, layer: 1, pos: 3361
type: A, layer: 1, pos: 3363
type: A, layer: 1, pos: 3364
type: A, layer: 1, pos: 3374
type: A, layer: 1, pos: 3554
type: A, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2965

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0161283, upper bound: 0.0161862
time: 12.34 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0161807, upper bound: 0.0161800
time: 194.49 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 213.05 seconds
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 213.05
Output dim: 1, lower bound: -0.0160045, upper bound: 0.0161859
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 213.05
Output dim: 1, lower bound: -0.0160569, upper bound: 0.0161900
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 213.05
Output dim: 1, lower bound: -0.0161279, upper bound: 0.0160815
NS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 213.05
Output dim: 1, lower bound: -0.0161801, upper bound: 0.0160779
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 213.05
Output dim: 1, lower bound: -0.0161283, upper bound: 0.0161862
NS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 213.05
Output dim: 1, lower bound: -0.0161807, upper bound: 0.0161800

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.8990098, -0.2410966, -0.8992416, -0.2410311, -0.3921038, 0.3922148
1: -0.0812790, 0.3244420, -0.0817973, 0.3246733, -0.0611911, 0.0574333
2: -2.2589509, -1.4729486, -2.2585270, -1.4728378, -0.2637583, 0.2635002
3: -2.8731174, -1.7067183, -2.8731785, -1.7066344, -0.4071909, 0.4071498
4: -3.5779858, -2.4292552, -3.5777872, -2.4291410, -0.4177418, 0.4176168
5: -2.5904109, -1.5439513, -2.5904677, -1.5438364, -0.2142418, 0.2141391
6: -2.5903931, -1.4590917, -2.5905721, -1.4587789, -0.1971295, 0.1967800
7: -2.8177888, -1.4407215, -2.8177259, -1.4406693, -0.5121695, 0.5113298
8: -0.0411508, 0.2033640, -0.0413072, 0.2033556, -0.1472202, 0.1475593
9: -0.8577836, -0.4629031, -0.8579589, -0.4630807, -0.2091658, 0.2078910

Time for backsubstitution: 8.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2966
type: B, layer: 1, pos: 3492
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2100
type: B, layer: 1, pos: 3108
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 2101
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2122
type: B, layer: 1, pos: 2333
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 345
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 3022
type: B, layer: 1, pos: 390
type: B, layer: 1, pos: 2221
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 2586
type: B, layer: 1, pos: 2703
type: B, layer: 1, pos: 2995
type: B, layer: 1, pos: 2710
type: B, layer: 1, pos: 2965
type: B, layer: 1, pos: 2701
type: B, layer: 1, pos: 2702
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 2700
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 380
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3421
type: B, layer: 1, pos: 2132
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 2120
type: B, layer: 1, pos: 2336
type: B, layer: 1, pos: 2865
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 2617
type: B, layer: 1, pos: 2097
type: B, layer: 1, pos: 2704
type: B, layer: 1, pos: 2705
type: B, layer: 1, pos: 3538
type: B, layer: 1, pos: 2850
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 2096
type: B, layer: 1, pos: 2375
type: B, layer: 1, pos: 377
type: B, layer: 1, pos: 379
type: B, layer: 1, pos: 2133
type: B, layer: 1, pos: 2321
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 3259
type: B, layer: 1, pos: 2600
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 376
type: B, layer: 1, pos: 2225
type: B, layer: 1, pos: 2820
type: B, layer: 1, pos: 3264
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 2862
type: B, layer: 1, pos: 2435
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 2994
type: B, layer: 1, pos: 2706
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2118
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2102
type: B, layer: 1, pos: 378
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2117
type: B, layer: 1, pos: 2709
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 274
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 3260
type: B, layer: 1, pos: 2860
type: B, layer: 1, pos: 3266
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 2891
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 2836
type: B, layer: 1, pos: 2855
type: B, layer: 1, pos: 2840
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 272
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 2116
type: B, layer: 1, pos: 3128
type: B, layer: 1, pos: 2651
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 2846
type: B, layer: 1, pos: 2858
type: B, layer: 1, pos: 3381
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3544
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 3529
type: B, layer: 1, pos: 2649
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 2859
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2666
type: B, layer: 1, pos: 433
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2232
type: B, layer: 1, pos: 3501
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 375
type: B, layer: 1, pos: 3383
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 3502
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 3343
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 509
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 2069
type: B, layer: 1, pos: 2099
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2129
type: B, layer: 1, pos: 2264
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2697
type: B, layer: 1, pos: 2864
type: B, layer: 1, pos: 3361
type: B, layer: 1, pos: 3363
type: B, layer: 1, pos: 3364
type: B, layer: 1, pos: 3374
type: B, layer: 1, pos: 3554
type: B, layer: 1, pos: 3597

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0159869, upper bound: 0.0160737
time: 7.48 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0159878, upper bound: 0.0161786
time: 67.70 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.8988650, -0.2409513, -0.8992622, -0.2409130, -0.3919834, 0.3922981
1: -0.0832011, 0.3234220, -0.0834876, 0.3246734, -0.0605691, 0.0609967
2: -2.2584436, -1.4724803, -2.2585330, -1.4724495, -0.2648153, 0.2633720
3: -2.8731680, -1.7068630, -2.8732224, -1.7066277, -0.4073369, 0.4070592
4: -3.5774906, -2.4289098, -3.5777967, -2.4288287, -0.4180582, 0.4178565
5: -2.5904269, -1.5440784, -2.5904813, -1.5438354, -0.2143288, 0.2140992
6: -2.5906949, -1.4593203, -2.5908177, -1.4587779, -0.1971948, 0.1971690
7: -2.8177118, -1.4405210, -2.8177471, -1.4405034, -0.5123695, 0.5115107
8: -0.0411172, 0.2033545, -0.0413299, 0.2033560, -0.1471652, 0.1475761
9: -0.8585107, -0.4635366, -0.8586024, -0.4630731, -0.2095213, 0.2084258

Time for backsubstitution: 6.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2966
type: B, layer: 1, pos: 3492
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2100
type: B, layer: 1, pos: 3108
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 2101
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2122
type: B, layer: 1, pos: 2333
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 345
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 3022
type: B, layer: 1, pos: 390
type: B, layer: 1, pos: 2221
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 2586
type: B, layer: 1, pos: 2703
type: B, layer: 1, pos: 2995
type: B, layer: 1, pos: 2965
type: B, layer: 1, pos: 2710
type: B, layer: 1, pos: 2701
type: B, layer: 1, pos: 2702
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 2700
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 380
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3421
type: B, layer: 1, pos: 2132
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 2120
type: B, layer: 1, pos: 2336
type: B, layer: 1, pos: 2865
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 2617
type: B, layer: 1, pos: 2097
type: B, layer: 1, pos: 2704
type: B, layer: 1, pos: 2705
type: B, layer: 1, pos: 3538
type: B, layer: 1, pos: 2850
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 2096
type: B, layer: 1, pos: 2375
type: B, layer: 1, pos: 377
type: B, layer: 1, pos: 379
type: B, layer: 1, pos: 2133
type: B, layer: 1, pos: 2321
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 3259
type: B, layer: 1, pos: 2600
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 376
type: B, layer: 1, pos: 2225
type: B, layer: 1, pos: 2820
type: B, layer: 1, pos: 3264
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 2862
type: B, layer: 1, pos: 2435
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 2994
type: B, layer: 1, pos: 2706
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2118
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2102
type: B, layer: 1, pos: 378
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2117
type: B, layer: 1, pos: 2709
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 274
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 3260
type: B, layer: 1, pos: 2860
type: B, layer: 1, pos: 3266
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 2891
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 2836
type: B, layer: 1, pos: 2855
type: B, layer: 1, pos: 2840
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 272
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 2116
type: B, layer: 1, pos: 3128
type: B, layer: 1, pos: 2651
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 2846
type: B, layer: 1, pos: 2858
type: B, layer: 1, pos: 3381
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3544
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 3529
type: B, layer: 1, pos: 2649
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 2859
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2666
type: B, layer: 1, pos: 433
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2232
type: B, layer: 1, pos: 3501
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 375
type: B, layer: 1, pos: 3383
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 3502
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 3343
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 509
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 2069
type: B, layer: 1, pos: 2099
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2129
type: B, layer: 1, pos: 2264
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2697
type: B, layer: 1, pos: 2864
type: B, layer: 1, pos: 3361
type: B, layer: 1, pos: 3363
type: B, layer: 1, pos: 3364
type: B, layer: 1, pos: 3374
type: B, layer: 1, pos: 3554
type: B, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0160395, upper bound: 0.0160696
time: 13.79 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0160387, upper bound: 0.0161976
time: 3.85 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.8994551, -0.2408253, -0.8992903, -0.2407948, -0.3927528, 0.3924008
1: -0.0832461, 0.3256933, -0.0835202, 0.3246734, -0.0599540, 0.0605763
2: -2.2590468, -1.4727914, -2.2585316, -1.4727017, -0.2637964, 0.2635325
3: -2.8735251, -1.7064611, -2.8735397, -1.7066286, -0.4069529, 0.4077839
4: -3.5783179, -2.4285669, -3.5778131, -2.4285092, -0.4188857, 0.4173777
5: -2.5908341, -1.5437000, -2.5908437, -1.5438304, -0.2140084, 0.2147953
6: -2.5912342, -1.4585502, -2.5913045, -1.4587784, -0.1966395, 0.1980510
7: -2.8178415, -1.4406098, -2.8177428, -1.4405708, -0.5123208, 0.5113835
8: -0.0413800, 0.2033671, -0.0413236, 0.2033584, -0.1474475, 0.1475798
9: -0.8585552, -0.4624364, -0.8586531, -0.4630772, -0.2086811, 0.2091385

Time for backsubstitution: 6.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2966
type: B, layer: 1, pos: 3492
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2100
type: B, layer: 1, pos: 3108
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 2101
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2122
type: B, layer: 1, pos: 2333
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 345
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 3022
type: B, layer: 1, pos: 390
type: B, layer: 1, pos: 2221
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 2586
type: B, layer: 1, pos: 2703
type: B, layer: 1, pos: 2995
type: B, layer: 1, pos: 2710
type: B, layer: 1, pos: 2965
type: B, layer: 1, pos: 2701
type: B, layer: 1, pos: 2702
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 2700
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 380
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3421
type: B, layer: 1, pos: 2132
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 2120
type: B, layer: 1, pos: 2336
type: B, layer: 1, pos: 2865
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 2617
type: B, layer: 1, pos: 2097
type: B, layer: 1, pos: 2704
type: B, layer: 1, pos: 2705
type: B, layer: 1, pos: 3538
type: B, layer: 1, pos: 2850
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 2096
type: B, layer: 1, pos: 2375
type: B, layer: 1, pos: 377
type: B, layer: 1, pos: 379
type: B, layer: 1, pos: 2133
type: B, layer: 1, pos: 2321
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 3259
type: B, layer: 1, pos: 2600
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 376
type: B, layer: 1, pos: 2225
type: B, layer: 1, pos: 2820
type: B, layer: 1, pos: 3264
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 2862
type: B, layer: 1, pos: 2435
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 2994
type: B, layer: 1, pos: 2706
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2118
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2102
type: B, layer: 1, pos: 378
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2117
type: B, layer: 1, pos: 2709
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 274
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 3260
type: B, layer: 1, pos: 2860
type: B, layer: 1, pos: 3266
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 2891
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 2836
type: B, layer: 1, pos: 2855
type: B, layer: 1, pos: 2840
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 272
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 2116
type: B, layer: 1, pos: 3128
type: B, layer: 1, pos: 2651
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 2846
type: B, layer: 1, pos: 2858
type: B, layer: 1, pos: 3381
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3544
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 3529
type: B, layer: 1, pos: 2649
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 2859
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2666
type: B, layer: 1, pos: 433
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2232
type: B, layer: 1, pos: 3501
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 375
type: B, layer: 1, pos: 3383
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 3502
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 3343
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 509
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 2069
type: B, layer: 1, pos: 2099
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2129
type: B, layer: 1, pos: 2264
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2697
type: B, layer: 1, pos: 2864
type: B, layer: 1, pos: 3361
type: B, layer: 1, pos: 3363
type: B, layer: 1, pos: 3364
type: B, layer: 1, pos: 3374
type: B, layer: 1, pos: 3554
type: B, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0161275, upper bound: 0.0160735
time: 3.98 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0161280, upper bound: 0.0161939
time: 3.58 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 13.66 seconds
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 13.66
Output dim: 1, lower bound: -0.0159869, upper bound: 0.0160737
NS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 13.66
Output dim: 1, lower bound: -0.0159878, upper bound: 0.0161786
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 13.66
Output dim: 1, lower bound: -0.0160395, upper bound: 0.0160696
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 13.66
Output dim: 1, lower bound: -0.0160387, upper bound: 0.0161976
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 13.66
Output dim: 1, lower bound: -0.0161275, upper bound: 0.0160735
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 13.66
Output dim: 1, lower bound: -0.0161280, upper bound: 0.0161939

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.8988627, -0.2409546, -0.8995957, -0.2409163, -0.3919307, 0.3926155
1: -0.0831718, 0.3234220, -0.0834557, 0.3258125, -0.0617626, 0.0600817
2: -2.2584410, -1.4724824, -2.2586250, -1.4724349, -0.2647892, 0.2633857
3: -2.8731570, -1.7068641, -2.8732095, -1.7063797, -0.4075887, 0.4068381
4: -3.5774894, -2.4289539, -3.5781476, -2.4288495, -0.4177729, 0.4182433
5: -2.5904205, -1.5440791, -2.5904732, -1.5435596, -0.2146119, 0.2138749
6: -2.5906827, -1.4593196, -2.5908122, -1.4582040, -0.1977534, 0.1967348
7: -2.8177099, -1.4405708, -2.8178163, -1.4405550, -0.5123176, 0.5115390
8: -0.0411162, 0.2033510, -0.0415153, 0.2033521, -0.1471500, 0.1477450
9: -0.8584964, -0.4635371, -0.8585908, -0.4627185, -0.2099010, 0.2081035

Time for backsubstitution: 6.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2966
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2100
type: A, layer: 1, pos: 3108
type: A, layer: 1, pos: 2431
type: A, layer: 1, pos: 2101
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 2122
type: A, layer: 1, pos: 2333
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 345
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 3022
type: A, layer: 1, pos: 390
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 2703
type: A, layer: 1, pos: 2995
type: A, layer: 1, pos: 2710
type: A, layer: 1, pos: 2701
type: A, layer: 1, pos: 2702
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 2700
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 380
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3421
type: A, layer: 1, pos: 3106
type: A, layer: 1, pos: 2132
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 2120
type: A, layer: 1, pos: 2336
type: A, layer: 1, pos: 2865
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 2097
type: A, layer: 1, pos: 2704
type: A, layer: 1, pos: 2705
type: A, layer: 1, pos: 3538
type: A, layer: 1, pos: 2850
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 2096
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 377
type: A, layer: 1, pos: 379
type: A, layer: 1, pos: 2133
type: A, layer: 1, pos: 2321
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 3259
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 376
type: A, layer: 1, pos: 2225
type: A, layer: 1, pos: 2820
type: A, layer: 1, pos: 3264
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 2862
type: A, layer: 1, pos: 2435
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 2994
type: A, layer: 1, pos: 2706
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2118
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2102
type: A, layer: 1, pos: 378
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 2709
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 274
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 3260
type: A, layer: 1, pos: 2860
type: A, layer: 1, pos: 3266
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 2891
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 2836
type: A, layer: 1, pos: 2855
type: A, layer: 1, pos: 2840
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 272
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 2116
type: A, layer: 1, pos: 3128
type: A, layer: 1, pos: 2651
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 2846
type: A, layer: 1, pos: 2858
type: A, layer: 1, pos: 3381
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3544
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 3529
type: A, layer: 1, pos: 2649
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2859
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 2413
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 433
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 3501
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 375
type: A, layer: 1, pos: 3383
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 3502
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 3343
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 509
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2099
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2129
type: A, layer: 1, pos: 2264
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2697
type: A, layer: 1, pos: 2864
type: A, layer: 1, pos: 3361
type: A, layer: 1, pos: 3363
type: A, layer: 1, pos: 3364
type: A, layer: 1, pos: 3374
type: A, layer: 1, pos: 3554
type: A, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2966

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0159493, upper bound: 0.0161527
time: 11.26 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0160029, upper bound: 0.0161475
time: 9.79 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.8994531, -0.2408283, -0.8996238, -0.2407975, -0.3927007, 0.3927186
1: -0.0832208, 0.3256932, -0.0834920, 0.3258126, -0.0611440, 0.0596609
2: -2.2590442, -1.4727929, -2.2586236, -1.4726863, -0.2637710, 0.2635208
3: -2.8735154, -1.7064617, -2.8735294, -1.7063807, -0.4072002, 0.4075518
4: -3.5783174, -2.4285927, -3.5781658, -2.4285269, -0.4186007, 0.4177651
5: -2.5908282, -1.5437005, -2.5908368, -1.5435548, -0.2142912, 0.2145709
6: -2.5912242, -1.4585502, -2.5913002, -1.4582043, -0.1971968, 0.1976169
7: -2.8178394, -1.4406345, -2.8178115, -1.4405966, -0.5122817, 0.5114342
8: -0.0413795, 0.2033637, -0.0415096, 0.2033544, -0.1474328, 0.1477492
9: -0.8585439, -0.4624370, -0.8586442, -0.4627229, -0.2090392, 0.2088095

Time for backsubstitution: 6.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2966
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2100
type: A, layer: 1, pos: 3108
type: A, layer: 1, pos: 2431
type: A, layer: 1, pos: 2101
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 2122
type: A, layer: 1, pos: 2333
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 345
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 3022
type: A, layer: 1, pos: 390
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 2703
type: A, layer: 1, pos: 2995
type: A, layer: 1, pos: 2710
type: A, layer: 1, pos: 2701
type: A, layer: 1, pos: 2702
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 2700
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 380
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3421
type: A, layer: 1, pos: 3106
type: A, layer: 1, pos: 2132
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 2120
type: A, layer: 1, pos: 2336
type: A, layer: 1, pos: 2865
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 2097
type: A, layer: 1, pos: 2704
type: A, layer: 1, pos: 2705
type: A, layer: 1, pos: 3538
type: A, layer: 1, pos: 2850
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 2096
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 377
type: A, layer: 1, pos: 379
type: A, layer: 1, pos: 2133
type: A, layer: 1, pos: 2321
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 3259
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 376
type: A, layer: 1, pos: 2225
type: A, layer: 1, pos: 2820
type: A, layer: 1, pos: 3264
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 2862
type: A, layer: 1, pos: 2435
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 2994
type: A, layer: 1, pos: 2706
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2118
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2102
type: A, layer: 1, pos: 378
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 2709
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 274
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 3260
type: A, layer: 1, pos: 2860
type: A, layer: 1, pos: 3266
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 2891
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 2836
type: A, layer: 1, pos: 2855
type: A, layer: 1, pos: 2840
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 272
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 2116
type: A, layer: 1, pos: 3128
type: A, layer: 1, pos: 2651
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 2846
type: A, layer: 1, pos: 2858
type: A, layer: 1, pos: 3381
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3544
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 3529
type: A, layer: 1, pos: 2649
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2859
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 2413
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 433
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 3501
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 375
type: A, layer: 1, pos: 3383
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 3502
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 3343
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 509
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2099
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2129
type: A, layer: 1, pos: 2264
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2697
type: A, layer: 1, pos: 2864
type: A, layer: 1, pos: 3361
type: A, layer: 1, pos: 3363
type: A, layer: 1, pos: 3364
type: A, layer: 1, pos: 3374
type: A, layer: 1, pos: 3554
type: A, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2966

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3492

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0160333, upper bound: 0.0161771
time: 30.75 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0161269, upper bound: 0.0161888
time: 4.80 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 44.60 seconds
NS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 44.60
Output dim: 1, lower bound: -0.0159493, upper bound: 0.0161527
NS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 44.60
Output dim: 1, lower bound: -0.0160029, upper bound: 0.0161475
NS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 44.60
Output dim: 1, lower bound: -0.0160333, upper bound: 0.0161771
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 44.60
Output dim: 1, lower bound: -0.0161269, upper bound: 0.0161888

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.8999604, -0.2408248, -0.8996148, -0.2408383, -0.3933888, 0.3925812
1: -0.0832170, 0.3268647, -0.0834865, 0.3258126, -0.0603449, 0.0607540
2: -2.2609076, -1.4721644, -2.2586226, -1.4726934, -0.2656221, 0.2629142
3: -2.8754826, -1.7063236, -2.8735280, -1.7063892, -0.4091915, 0.4067159
4: -3.5784230, -2.4271071, -3.5781596, -2.4285271, -0.4180478, 0.4192497
5: -2.5921218, -1.5436795, -2.5908365, -1.5435612, -0.2155014, 0.2137340
6: -2.5920420, -1.4585304, -2.5913000, -1.4582076, -0.1979782, 0.1972676
7: -2.8200159, -1.4406272, -2.8178117, -1.4406028, -0.5143230, 0.5107507
8: -0.0416239, 0.2037868, -0.0414977, 0.2033544, -0.1474600, 0.1481623
9: -0.8590034, -0.4623832, -0.8586433, -0.4627291, -0.2095648, 0.2088623

Time for backsubstitution: 6.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2966
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2100
type: B, layer: 1, pos: 3108
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 2101
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2122
type: B, layer: 1, pos: 2333
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 345
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 3022
type: B, layer: 1, pos: 390
type: B, layer: 1, pos: 2221
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 2586
type: B, layer: 1, pos: 2703
type: B, layer: 1, pos: 2995
type: B, layer: 1, pos: 2710
type: B, layer: 1, pos: 2965
type: B, layer: 1, pos: 2701
type: B, layer: 1, pos: 2702
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 2700
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 380
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3421
type: B, layer: 1, pos: 2132
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 2120
type: B, layer: 1, pos: 2336
type: B, layer: 1, pos: 2865
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 2617
type: B, layer: 1, pos: 3492
type: B, layer: 1, pos: 2097
type: B, layer: 1, pos: 2704
type: B, layer: 1, pos: 2705
type: B, layer: 1, pos: 3538
type: B, layer: 1, pos: 2850
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 2096
type: B, layer: 1, pos: 2375
type: B, layer: 1, pos: 377
type: B, layer: 1, pos: 379
type: B, layer: 1, pos: 2133
type: B, layer: 1, pos: 2321
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 3259
type: B, layer: 1, pos: 2600
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 376
type: B, layer: 1, pos: 2225
type: B, layer: 1, pos: 2820
type: B, layer: 1, pos: 3264
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 2862
type: B, layer: 1, pos: 2435
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 2994
type: B, layer: 1, pos: 2706
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2118
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2102
type: B, layer: 1, pos: 378
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2117
type: B, layer: 1, pos: 2709
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 274
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 3260
type: B, layer: 1, pos: 2860
type: B, layer: 1, pos: 3266
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 2891
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 2836
type: B, layer: 1, pos: 2855
type: B, layer: 1, pos: 2840
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 272
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 2116
type: B, layer: 1, pos: 3128
type: B, layer: 1, pos: 2651
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 2846
type: B, layer: 1, pos: 2858
type: B, layer: 1, pos: 3381
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3544
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 3529
type: B, layer: 1, pos: 2649
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 2859
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2666
type: B, layer: 1, pos: 433
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2232
type: B, layer: 1, pos: 3501
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 375
type: B, layer: 1, pos: 3383
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 3502
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 3343
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 509
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 2069
type: B, layer: 1, pos: 2099
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2129
type: B, layer: 1, pos: 2264
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2697
type: B, layer: 1, pos: 2864
type: B, layer: 1, pos: 3361
type: B, layer: 1, pos: 3363
type: B, layer: 1, pos: 3364
type: B, layer: 1, pos: 3374
type: B, layer: 1, pos: 3554
type: B, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2966

### Candidate
type: B, layer: 1, pos: 2430

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0161118, upper bound: 0.0161452
time: 3.80 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0161109, upper bound: 0.0160607
time: 114.31 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 124.51 seconds
NS_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 124.51
Output dim: 1, lower bound: -0.0161118, upper bound: 0.0161452
NS_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 124.51
Output dim: 1, lower bound: -0.0161109, upper bound: 0.0160607

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 36.17 + 1262.29 = 1298.46 seconds
