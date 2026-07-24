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
execution time: IAR + RelationalAnalysis = 8.42 + 186.01 = 194.43 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0578099, upper bound: 0.0578111

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 504
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 291
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3375
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 248
type: DSZ, layer: 1, pos: 2662
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3581
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 3325
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 3398
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 2750
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 3549
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 404
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2624

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2326

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0578080, upper bound: 0.0577913
time: 229.15 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0577894, upper bound: 0.0578014
time: 212.51 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 441.67 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 441.67
Output dim: 9, lower bound: -0.0578080, upper bound: 0.0577913
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 441.67
Output dim: 9, lower bound: -0.0577894, upper bound: 0.0578014

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -3.1605945, -1.7897048, -3.1605945, -1.7897048, -0.9697499, 0.9697239
1: -0.3287182, 0.6209364, -0.3287182, 0.6209364, -0.8802981, 0.8802977
2: -1.9960757, -1.5942692, -1.9960757, -1.5942692, -0.1939666, 0.1939754
3: -1.4003946, -0.5967301, -1.4003946, -0.5967301, -0.3756310, 0.3756292
4: -2.4694018, -1.8323946, -2.4694018, -1.8323946, -0.2926504, 0.2926484
5: -1.8780696, -0.9551735, -1.8780696, -0.9551735, -0.4068788, 0.4068764
6: -1.9722595, -1.2424653, -1.9722595, -1.2424653, -0.2244010, 0.2244034
7: -1.1080118, -0.6143922, -1.1080118, -0.6143922, -0.2989789, 0.2989793
8: -2.8991513, -1.5445766, -2.8991513, -1.5445766, -1.1594133, 1.1593843
9: 0.4037505, 1.0025887, 0.4037505, 1.0025887, -0.5704978, 0.5704975

Time for backsubstitution: 6.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 3549
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 504
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 3398
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 248
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 3581
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2750
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 404
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 3375
type: DSZ, layer: 1, pos: 291
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2662
type: DSZ, layer: 1, pos: 3325
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2272

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2238

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0578017, upper bound: 0.0577913
time: 17.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0578017, upper bound: 0.0577911
time: 15.04 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -3.1605945, -1.7897048, -3.1605945, -1.7897048, -0.9697237, 0.9697497
1: -0.3287182, 0.6209364, -0.3287182, 0.6209364, -0.8802979, 0.8802986
2: -1.9960757, -1.5942692, -1.9960757, -1.5942692, -0.1939755, 0.1939666
3: -1.4003946, -0.5967301, -1.4003946, -0.5967301, -0.3756292, 0.3756310
4: -2.4694018, -1.8323946, -2.4694018, -1.8323946, -0.2926484, 0.2926504
5: -1.8780696, -0.9551735, -1.8780696, -0.9551735, -0.4068764, 0.4068788
6: -1.9722595, -1.2424653, -1.9722595, -1.2424653, -0.2244034, 0.2244010
7: -1.1080118, -0.6143922, -1.1080118, -0.6143922, -0.2989792, 0.2989789
8: -2.8991513, -1.5445766, -2.8991513, -1.5445766, -1.1593843, 1.1594138
9: 0.4037505, 1.0025887, 0.4037505, 1.0025887, -0.5704975, 0.5704980

Time for backsubstitution: 6.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 404
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 248
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 2662
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 3398
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 3581
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2750
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 504
type: DSZ, layer: 1, pos: 3549
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 291
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3375
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 3325
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2550

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0577802, upper bound: 0.0577898
time: 407.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0577793, upper bound: 0.0577919
time: 228.14 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 641.70 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 641.70
Output dim: 9, lower bound: -0.0578017, upper bound: 0.0577913
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 641.70
Output dim: 9, lower bound: -0.0578017, upper bound: 0.0577911
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 641.70
Output dim: 9, lower bound: -0.0577802, upper bound: 0.0577898
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 641.70
Output dim: 9, lower bound: -0.0577793, upper bound: 0.0577919

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.1605945, -1.7897048, -3.1605945, -1.7897048, -0.9697499, 0.9697239
1: -0.3287182, 0.6209364, -0.3287182, 0.6209364, -0.8802981, 0.8802977
2: -1.9960757, -1.5942692, -1.9960757, -1.5942692, -0.1939666, 0.1939754
3: -1.4003946, -0.5967301, -1.4003946, -0.5967301, -0.3756310, 0.3756292
4: -2.4694018, -1.8323946, -2.4694018, -1.8323946, -0.2926504, 0.2926484
5: -1.8780696, -0.9551735, -1.8780696, -0.9551735, -0.4068788, 0.4068764
6: -1.9722595, -1.2424653, -1.9722595, -1.2424653, -0.2244010, 0.2244034
7: -1.1080118, -0.6143922, -1.1080118, -0.6143922, -0.2989789, 0.2989793
8: -2.8991513, -1.5445766, -2.8991513, -1.5445766, -1.1594133, 1.1593843
9: 0.4037505, 1.0025887, 0.4037505, 1.0025887, -0.5704978, 0.5704975

Time for backsubstitution: 6.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 248
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2750
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 3581
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 3549
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 3375
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 404
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 3398
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2662
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3325
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 504
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 291
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2325

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0578005, upper bound: 0.0577865
time: 9.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0577963, upper bound: 0.0577893
time: 161.99 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.1605945, -1.7897048, -3.1605945, -1.7897048, -0.9697499, 0.9697239
1: -0.3287182, 0.6209364, -0.3287182, 0.6209364, -0.8802981, 0.8802977
2: -1.9960757, -1.5942692, -1.9960757, -1.5942692, -0.1939666, 0.1939754
3: -1.4003946, -0.5967301, -1.4003946, -0.5967301, -0.3756310, 0.3756292
4: -2.4694018, -1.8323946, -2.4694018, -1.8323946, -0.2926504, 0.2926484
5: -1.8780696, -0.9551735, -1.8780696, -0.9551735, -0.4068788, 0.4068764
6: -1.9722595, -1.2424653, -1.9722595, -1.2424653, -0.2244010, 0.2244034
7: -1.1080118, -0.6143922, -1.1080118, -0.6143922, -0.2989789, 0.2989793
8: -2.8991513, -1.5445766, -2.8991513, -1.5445766, -1.1594133, 1.1593843
9: 0.4037505, 1.0025887, 0.4037505, 1.0025887, -0.5704978, 0.5704975

Time for backsubstitution: 6.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2750
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 248
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 404
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 3581
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 3325
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 291
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2662
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 3398
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 3375
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 504
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 3549
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 3029

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3280

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0577736, upper bound: 0.0577702
time: 81.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0577813, upper bound: 0.0577605
time: 184.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.1605945, -1.7897048, -3.1605945, -1.7897048, -0.9695926, 0.9696198
1: -0.3287182, 0.6209364, -0.3287182, 0.6209364, -0.8802965, 0.8802969
2: -1.9960757, -1.5942692, -1.9960757, -1.5942692, -0.1938806, 0.1938692
3: -1.4003946, -0.5967301, -1.4003946, -0.5967301, -0.3756076, 0.3756100
4: -2.4694018, -1.8323946, -2.4694018, -1.8323946, -0.2925910, 0.2925943
5: -1.8780696, -0.9551735, -1.8780696, -0.9551735, -0.4068355, 0.4068376
6: -1.9722595, -1.2424653, -1.9722595, -1.2424653, -0.2243913, 0.2243890
7: -1.1080118, -0.6143922, -1.1080118, -0.6143922, -0.2989773, 0.2989769
8: -2.8991513, -1.5445766, -2.8991513, -1.5445766, -1.1592221, 1.1592555
9: 0.4037505, 1.0025887, 0.4037505, 1.0025887, -0.5704894, 0.5704894

Time for backsubstitution: 6.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 3375
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 3581
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2750
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 404
type: DSZ, layer: 1, pos: 2662
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 3549
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 248
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 504
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 3325
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 3398
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 3279
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 291
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 101

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2364

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0577477, upper bound: 0.0577177
time: 454.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0577150, upper bound: 0.0577619
time: 13.77 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 475.06 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 475.06
Output dim: 9, lower bound: -0.0578005, upper bound: 0.0577865
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 475.06
Output dim: 9, lower bound: -0.0577963, upper bound: 0.0577893
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 475.06
Output dim: 9, lower bound: -0.0577736, upper bound: 0.0577702
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 475.06
Output dim: 9, lower bound: -0.0577813, upper bound: 0.0577605
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 475.06
Output dim: 9, lower bound: -0.0577477, upper bound: 0.0577177
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 475.06
Output dim: 9, lower bound: -0.0577150, upper bound: 0.0577619
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 475.06
Output dim: 9, lower bound: -0.0577793, upper bound: 0.0577919

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 194.43 + 2046.51 = 2240.94 seconds
