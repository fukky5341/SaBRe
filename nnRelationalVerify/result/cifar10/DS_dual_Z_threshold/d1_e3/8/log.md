## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 8)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.0577526895


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3.1605945, -1.7897048, -3.1605945, -1.7897048, -0.9704130, 0.9704132)
1: (-0.3287182, 0.6209364, -0.3287182, 0.6209364, -0.8803020, 0.8803022)
2: (-1.9960757, -1.5942692, -1.9960757, -1.5942692, -0.1947251, 0.1947251)
3: (-1.4003946, -0.5967301, -1.4003946, -0.5967301, -0.3757512, 0.3757513)
4: (-2.4694018, -1.8323946, -2.4694018, -1.8323946, -0.2932426, 0.2932426)
5: (-1.8780696, -0.9551735, -1.8780696, -0.9551735, -0.4072059, 0.4072059)
6: (-1.9722595, -1.2424653, -1.9722595, -1.2424653, -0.2245336, 0.2245337)
7: (-1.1080118, -0.6143922, -1.1080118, -0.6143922, -0.2989827, 0.2989827)
8: (-2.8991513, -1.5445766, -2.8991513, -1.5445766, -1.1609325, 1.1609321)
9: (0.4037505, 1.0025887, 0.4037505, 1.0025887, -0.5704436, 0.5704434)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.78 + 185.38 = 193.16 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0578099, upper bound: 0.0578111

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 3325
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 3581
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 3549
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 291
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2750
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 248
type: DSZ, layer: 1, pos: 404
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 504
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2662
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 3375
type: DSZ, layer: 1, pos: 3398

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3114

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0578051, upper bound: 0.0576134
time: 270.63 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0576179, upper bound: 0.0578011
time: 211.28 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 481.98 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 481.98
Output dim: 9, lower bound: -0.0578051, upper bound: 0.0576134
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 481.98
Output dim: 9, lower bound: -0.0576179, upper bound: 0.0578011

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -3.1605945, -1.7897048, -3.1605945, -1.7897048, -0.9707685, 0.9707842
1: -0.3287182, 0.6209364, -0.3287182, 0.6209364, -0.8803029, 0.8803029
2: -1.9960757, -1.5942692, -1.9960757, -1.5942692, -0.1946590, 0.1946605
3: -1.4003946, -0.5967301, -1.4003946, -0.5967301, -0.3756454, 0.3756459
4: -2.4694018, -1.8323946, -2.4694018, -1.8323946, -0.2897475, 0.2898483
5: -1.8780696, -0.9551735, -1.8780696, -0.9551735, -0.4070482, 0.4070501
6: -1.9722595, -1.2424653, -1.9722595, -1.2424653, -0.2242413, 0.2242430
7: -1.1080118, -0.6143922, -1.1080118, -0.6143922, -0.2988325, 0.2988365
8: -2.8991513, -1.5445766, -2.8991513, -1.5445766, -1.1609044, 1.1609039
9: 0.4037505, 1.0025887, 0.4037505, 1.0025887, -0.5704303, 0.5704303

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 3325
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 3581
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 3549
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 291
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2750
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 248
type: DSZ, layer: 1, pos: 404
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 504
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2662
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 3375
type: DSZ, layer: 1, pos: 3398

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2216

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0577856, upper bound: 0.0575686
time: 81.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0577546, upper bound: 0.0575985
time: 323.74 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -3.1605945, -1.7897048, -3.1605945, -1.7897048, -0.9707842, 0.9707689
1: -0.3287182, 0.6209364, -0.3287182, 0.6209364, -0.8803029, 0.8803029
2: -1.9960757, -1.5942692, -1.9960757, -1.5942692, -0.1946605, 0.1946591
3: -1.4003946, -0.5967301, -1.4003946, -0.5967301, -0.3756459, 0.3756454
4: -2.4694018, -1.8323946, -2.4694018, -1.8323946, -0.2898483, 0.2897475
5: -1.8780696, -0.9551735, -1.8780696, -0.9551735, -0.4070500, 0.4070483
6: -1.9722595, -1.2424653, -1.9722595, -1.2424653, -0.2242430, 0.2242413
7: -1.1080118, -0.6143922, -1.1080118, -0.6143922, -0.2988366, 0.2988325
8: -2.8991513, -1.5445766, -2.8991513, -1.5445766, -1.1609039, 1.1609049
9: 0.4037505, 1.0025887, 0.4037505, 1.0025887, -0.5704303, 0.5704303

Time for backsubstitution: 5.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 3325
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 3581
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 3549
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 291
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2750
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 248
type: DSZ, layer: 1, pos: 404
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 504
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2662
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 3375
type: DSZ, layer: 1, pos: 3398

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 2216

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0575973, upper bound: 0.0577557
time: 506.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0575674, upper bound: 0.0575989
time: 289.12 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 802.01 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 802.01
Output dim: 9, lower bound: -0.0577856, upper bound: 0.0575686
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 802.01
Output dim: 9, lower bound: -0.0577546, upper bound: 0.0575985
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 802.01
Output dim: 9, lower bound: -0.0575973, upper bound: 0.0577557
DS_DSZ2_DSZ2, status: Status.VERIFIED, split count: 2, time: 802.01
Output dim: 9, lower bound: -0.0575674, upper bound: 0.0575989

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 193.16 + 1695.01 = 1888.17 seconds
