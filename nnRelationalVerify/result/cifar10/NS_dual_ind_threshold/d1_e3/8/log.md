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
execution time: IAR + RelationalAnalysis = 7.89 + 185.75 = 193.64 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0578099, upper bound: 0.0578111

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3124
type: A, layer: 1, pos: 3094
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 2170
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 3112
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2437
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 2591
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2647
type: A, layer: 1, pos: 3026
type: A, layer: 1, pos: 2662
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2633
type: A, layer: 1, pos: 3095
type: A, layer: 1, pos: 2413
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 3061
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 3114
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 2570
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 2187
type: A, layer: 1, pos: 2391
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 2398
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 2611
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 2584
type: A, layer: 1, pos: 2432
type: A, layer: 1, pos: 2356
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 2823
type: A, layer: 1, pos: 2188
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 3040
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 3022
type: A, layer: 1, pos: 3096
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 2668
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 292
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 2680
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 3279
type: A, layer: 1, pos: 3549
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 248
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2550
type: A, layer: 1, pos: 2679
type: A, layer: 1, pos: 2326
type: A, layer: 1, pos: 3325
type: A, layer: 1, pos: 2538
type: A, layer: 1, pos: 2311
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 2764
type: A, layer: 1, pos: 3280
type: A, layer: 1, pos: 2315
type: A, layer: 1, pos: 385
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 2994
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 3217
type: A, layer: 1, pos: 2750
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 2595
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 2325
type: A, layer: 1, pos: 2972
type: A, layer: 1, pos: 3290
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 2979
type: A, layer: 1, pos: 458
type: A, layer: 1, pos: 3398
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2081
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 2987
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2724
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 2055
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 2067
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 2751
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 2068
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 2496
type: A, layer: 1, pos: 2070
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 2083
type: A, layer: 1, pos: 3581
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 2943
type: A, layer: 1, pos: 2250
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 291
type: A, layer: 1, pos: 2259
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 418
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 3375
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 504
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 3151
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 2251
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 2037
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 404
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2084
type: A, layer: 1, pos: 2204
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2242
type: A, layer: 1, pos: 2624
type: A, layer: 1, pos: 2669
type: A, layer: 1, pos: 2922
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3059
type: A, layer: 1, pos: 3373

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3124

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0576838, upper bound: 0.0573757
time: 702.23 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0578069, upper bound: 0.0578074
time: 373.32 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1075.63 seconds
NS_A1, status: Status.VERIFIED, split count: 1, time: 1075.63
Output dim: 9, lower bound: -0.0576838, upper bound: 0.0573757
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1075.63
Output dim: 9, lower bound: -0.0578069, upper bound: 0.0578074

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -3.1605945, -1.7897129, -3.1605945, -1.7897120, -0.9710121, 0.9699702
1: -0.3287158, 0.6208930, -0.3287160, 0.6208982, -0.8802752, 0.8802688
2: -1.9960425, -1.5942734, -1.9960468, -1.5942731, -0.1942829, 0.1946920
3: -1.4003733, -0.5967301, -1.4003761, -0.5967301, -0.3744623, 0.3757025
4: -2.4693198, -1.8323957, -2.4693303, -1.8323956, -0.2887508, 0.2931706
5: -1.8780540, -0.9551736, -1.8780559, -0.9551737, -0.4055644, 0.4071823
6: -1.9722528, -1.2424650, -1.9722540, -1.2424653, -0.2230469, 0.2245190
7: -1.1079758, -0.6143929, -1.1079800, -0.6143927, -0.2979380, 0.2989640
8: -2.8991499, -1.5446253, -2.8991508, -1.5446186, -1.1608896, 1.1607776
9: 0.4037508, 1.0025460, 0.4037508, 1.0025495, -0.5704203, 0.5703709

Time for backsubstitution: 5.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3094
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 2170
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 3112
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2437
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2591
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2647
type: B, layer: 1, pos: 3026
type: B, layer: 1, pos: 2662
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2633
type: B, layer: 1, pos: 3095
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 3061
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 3114
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 2570
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 2187
type: B, layer: 1, pos: 2391
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2398
type: B, layer: 1, pos: 2585
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 2600
type: B, layer: 1, pos: 2611
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 2584
type: B, layer: 1, pos: 2432
type: B, layer: 1, pos: 2356
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 2823
type: B, layer: 1, pos: 2188
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 3040
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 3022
type: B, layer: 1, pos: 3096
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 3124
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 292
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 2680
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 3549
type: B, layer: 1, pos: 2541
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 248
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2550
type: B, layer: 1, pos: 2679
type: B, layer: 1, pos: 2326
type: B, layer: 1, pos: 3325
type: B, layer: 1, pos: 2538
type: B, layer: 1, pos: 2311
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 2764
type: B, layer: 1, pos: 3280
type: B, layer: 1, pos: 2315
type: B, layer: 1, pos: 385
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 2994
type: B, layer: 1, pos: 2296
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 3217
type: B, layer: 1, pos: 2750
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 2595
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 2325
type: B, layer: 1, pos: 2972
type: B, layer: 1, pos: 3290
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 2979
type: B, layer: 1, pos: 458
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2081
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 2987
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2724
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 2055
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2067
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 2751
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 2068
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 2496
type: B, layer: 1, pos: 2070
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 2083
type: B, layer: 1, pos: 3581
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 2943
type: B, layer: 1, pos: 2250
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 291
type: B, layer: 1, pos: 2259
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 418
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 3375
type: B, layer: 1, pos: 2510
type: B, layer: 1, pos: 504
type: B, layer: 1, pos: 2051
type: B, layer: 1, pos: 3151
type: B, layer: 1, pos: 2052
type: B, layer: 1, pos: 2251
type: B, layer: 1, pos: 2036
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 2037
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 404
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2069
type: B, layer: 1, pos: 2084
type: B, layer: 1, pos: 2204
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2242
type: B, layer: 1, pos: 2624
type: B, layer: 1, pos: 2669
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3059
type: B, layer: 1, pos: 3373

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3094

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0575611, upper bound: 0.0576851
time: 213.19 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0577342, upper bound: 0.0577368
time: 7.71 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 226.88 seconds
NS_A2_B1, status: Status.VERIFIED, split count: 2, time: 226.88
Output dim: 9, lower bound: -0.0575611, upper bound: 0.0576851
NS_A2_B2, status: Status.VERIFIED, split count: 2, time: 226.88
Output dim: 9, lower bound: -0.0577342, upper bound: 0.0577368

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 193.64 + 1302.50 = 1496.15 seconds
