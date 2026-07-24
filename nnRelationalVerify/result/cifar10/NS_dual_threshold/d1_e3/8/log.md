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
execution time: IAR + RelationalAnalysis = 7.78 + 182.97 = 190.75 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0578099, upper bound: 0.0578111

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3094
type: A, layer: 1, pos: 3094
type: A, layer: 1, pos: 3124
type: B, layer: 1, pos: 3124
type: A, layer: 1, pos: 2170
type: B, layer: 1, pos: 2170
type: A, layer: 1, pos: 2597
type: B, layer: 1, pos: 2597
type: A, layer: 1, pos: 2372
type: B, layer: 1, pos: 2372
type: A, layer: 1, pos: 2156
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2390
type: A, layer: 1, pos: 2390
type: B, layer: 1, pos: 3112
type: A, layer: 1, pos: 3112
type: B, layer: 1, pos: 2591
type: A, layer: 1, pos: 2591
type: B, layer: 1, pos: 3055
type: A, layer: 1, pos: 3055
type: B, layer: 1, pos: 2437
type: A, layer: 1, pos: 2437
type: A, layer: 1, pos: 2405
type: B, layer: 1, pos: 2405
type: A, layer: 1, pos: 2647
type: B, layer: 1, pos: 2647
type: A, layer: 1, pos: 3026
type: B, layer: 1, pos: 3026
type: B, layer: 1, pos: 2423
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 2662
type: B, layer: 1, pos: 2662
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 3095
type: B, layer: 1, pos: 3095
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2386
type: A, layer: 1, pos: 2386
type: B, layer: 1, pos: 3061
type: A, layer: 1, pos: 3061
type: B, layer: 1, pos: 2659
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2364
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 2413
type: A, layer: 1, pos: 2413
type: B, layer: 1, pos: 2570
type: A, layer: 1, pos: 2570
type: A, layer: 1, pos: 2379
type: B, layer: 1, pos: 2379
type: A, layer: 1, pos: 2391
type: B, layer: 1, pos: 2391
type: A, layer: 1, pos: 2633
type: B, layer: 1, pos: 2633
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 2398
type: A, layer: 1, pos: 2398
type: A, layer: 1, pos: 2585
type: B, layer: 1, pos: 2585
type: B, layer: 1, pos: 2600
type: A, layer: 1, pos: 2600
type: B, layer: 1, pos: 2611
type: A, layer: 1, pos: 2611
type: A, layer: 1, pos: 2584
type: B, layer: 1, pos: 2584
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 2356
type: B, layer: 1, pos: 2356
type: A, layer: 1, pos: 2432
type: B, layer: 1, pos: 2432
type: B, layer: 1, pos: 2187
type: A, layer: 1, pos: 2187
type: A, layer: 1, pos: 2823
type: B, layer: 1, pos: 2823
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 2583
type: B, layer: 1, pos: 2583
type: A, layer: 1, pos: 3022
type: B, layer: 1, pos: 3022
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3114
type: A, layer: 1, pos: 3114
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 3040
type: B, layer: 1, pos: 3040
type: A, layer: 1, pos: 2188
type: B, layer: 1, pos: 2188
type: B, layer: 1, pos: 2422
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 2193
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 3096
type: A, layer: 1, pos: 3096
type: A, layer: 1, pos: 248
type: B, layer: 1, pos: 248
type: A, layer: 1, pos: 3274
type: B, layer: 1, pos: 3274
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 2542
type: B, layer: 1, pos: 2542
type: A, layer: 1, pos: 816
type: B, layer: 1, pos: 816
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 292
type: B, layer: 1, pos: 292
type: A, layer: 1, pos: 3279
type: B, layer: 1, pos: 3279
type: A, layer: 1, pos: 3549
type: B, layer: 1, pos: 3549
type: A, layer: 1, pos: 2541
type: B, layer: 1, pos: 2541
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 291
type: B, layer: 1, pos: 291
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2764
type: A, layer: 1, pos: 2764
type: A, layer: 1, pos: 2550
type: B, layer: 1, pos: 2550
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 2750
type: B, layer: 1, pos: 2750
type: B, layer: 1, pos: 2326
type: A, layer: 1, pos: 2326
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2538
type: A, layer: 1, pos: 2538
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 2315
type: B, layer: 1, pos: 2315
type: A, layer: 1, pos: 2994
type: B, layer: 1, pos: 2994
type: A, layer: 1, pos: 2311
type: B, layer: 1, pos: 2311
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 458
type: B, layer: 1, pos: 458
type: A, layer: 1, pos: 3217
type: B, layer: 1, pos: 3217
type: A, layer: 1, pos: 3280
type: B, layer: 1, pos: 3280
type: B, layer: 1, pos: 2201
type: A, layer: 1, pos: 2201
type: B, layer: 1, pos: 2663
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 3398
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2979
type: A, layer: 1, pos: 2979
type: A, layer: 1, pos: 3290
type: B, layer: 1, pos: 3290
type: A, layer: 1, pos: 2229
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2296
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2522
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 2304
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 2329
type: A, layer: 1, pos: 2329
type: B, layer: 1, pos: 2325
type: A, layer: 1, pos: 2325
type: A, layer: 1, pos: 2081
type: B, layer: 1, pos: 2081
type: A, layer: 1, pos: 2679
type: B, layer: 1, pos: 2679
type: A, layer: 1, pos: 2066
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 2595
type: A, layer: 1, pos: 2595
type: A, layer: 1, pos: 2615
type: B, layer: 1, pos: 2615
type: A, layer: 1, pos: 2724
type: B, layer: 1, pos: 2724
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 385
type: B, layer: 1, pos: 385
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2526
type: A, layer: 1, pos: 2526
type: B, layer: 1, pos: 2972
type: A, layer: 1, pos: 2972
type: B, layer: 1, pos: 2987
type: A, layer: 1, pos: 2987
type: B, layer: 1, pos: 2272
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 2067
type: B, layer: 1, pos: 2067
type: A, layer: 1, pos: 720
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 2962
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 2751
type: B, layer: 1, pos: 2751
type: A, layer: 1, pos: 2330
type: B, layer: 1, pos: 2330
type: A, layer: 1, pos: 3581
type: B, layer: 1, pos: 3581
type: B, layer: 1, pos: 418
type: A, layer: 1, pos: 418
type: B, layer: 1, pos: 2680
type: A, layer: 1, pos: 2680
type: B, layer: 1, pos: 2068
type: A, layer: 1, pos: 2068
type: A, layer: 1, pos: 2055
type: B, layer: 1, pos: 2055
type: A, layer: 1, pos: 2316
type: B, layer: 1, pos: 2316
type: A, layer: 1, pos: 2083
type: B, layer: 1, pos: 2083
type: B, layer: 1, pos: 2496
type: A, layer: 1, pos: 2496
type: A, layer: 1, pos: 2668
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 2289
type: A, layer: 1, pos: 2289
type: B, layer: 1, pos: 3325
type: A, layer: 1, pos: 3325
type: B, layer: 1, pos: 3427
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 2510
type: B, layer: 1, pos: 2510
type: B, layer: 1, pos: 2964
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 2216
type: B, layer: 1, pos: 2216
type: A, layer: 1, pos: 2250
type: B, layer: 1, pos: 2250
type: A, layer: 1, pos: 2943
type: B, layer: 1, pos: 2943
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2070
type: A, layer: 1, pos: 2070
type: A, layer: 1, pos: 2259
type: B, layer: 1, pos: 2259
type: B, layer: 1, pos: 2527
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 3375
type: B, layer: 1, pos: 3375
type: B, layer: 1, pos: 2947
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 2051
type: B, layer: 1, pos: 2051
type: A, layer: 1, pos: 504
type: B, layer: 1, pos: 504
type: B, layer: 1, pos: 3151
type: A, layer: 1, pos: 3151
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 2052
type: B, layer: 1, pos: 2052
type: A, layer: 1, pos: 2036
type: B, layer: 1, pos: 2036
type: A, layer: 1, pos: 2053
type: B, layer: 1, pos: 2053
type: A, layer: 1, pos: 2038
type: B, layer: 1, pos: 2038
type: A, layer: 1, pos: 2037
type: B, layer: 1, pos: 2037
type: B, layer: 1, pos: 2300
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 2251
type: A, layer: 1, pos: 2251
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 692
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

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3094

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0575632, upper bound: 0.0576891
time: 695.28 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0575705, upper bound: 0.0577412
time: 27.50 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 722.86 seconds
NS_B1, status: Status.VERIFIED, split count: 1, time: 722.86
Output dim: 9, lower bound: -0.0575632, upper bound: 0.0576891
NS_B2, status: Status.VERIFIED, split count: 1, time: 722.86
Output dim: 9, lower bound: -0.0575705, upper bound: 0.0577412

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 190.75 + 722.86 = 913.61 seconds
