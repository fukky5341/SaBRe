## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 6)
Time budget: 3600 seconds
Split limit: 100
Threshold: 0.21667091219999998


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3.5651972, -1.4679933, -3.5651972, -1.4679933, -1.7980355, 1.7980351)
1: (-1.7701428, 0.2758193, -1.7701428, 0.2758193, -1.8970647, 1.8970649)
2: (-1.5424505, -0.9444594, -1.5424505, -0.9444594, -0.1907168, 0.1907169)
3: (-0.9903108, -0.2777686, -0.9903108, -0.2777686, -0.4601189, 0.4601189)
4: (-0.7439071, -0.2672053, -0.7439071, -0.2672053, -0.2307192, 0.2307192)
5: (-0.9715933, -0.1104509, -0.9715933, -0.1104509, -0.4552237, 0.4552236)
6: (-0.8177381, -0.3731477, -0.8177381, -0.3731477, -0.1619204, 0.1619204)
7: (-0.7450799, 0.0228758, -0.7450799, 0.0228758, -0.5879859, 0.5879859)
8: (-3.9531717, -2.1230025, -3.9531717, -2.1230025, -0.9441488, 0.9441488)
9: (-1.9562843, -0.3158286, -1.9562843, -0.3158286, -1.2892140, 1.2892141)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 8.06 + 674.93 = 682.99 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.2168844, upper bound: 0.2168864

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2363
type: A, layer: 1, pos: 2362
type: A, layer: 1, pos: 2364
type: A, layer: 1, pos: 2570
type: A, layer: 1, pos: 2599
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 262
type: A, layer: 1, pos: 2125
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 100
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 2331
type: A, layer: 1, pos: 2336
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 2170
type: A, layer: 1, pos: 347
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 287
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 3040
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 403
type: A, layer: 1, pos: 2094
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 3567
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2419
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 2591
type: A, layer: 1, pos: 2553
type: A, layer: 1, pos: 2607
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 317
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 378
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 3459
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 263
type: A, layer: 1, pos: 2132
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 3013
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 3273
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 3112
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 3125
type: A, layer: 1, pos: 387
type: A, layer: 1, pos: 2507
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 2322
type: A, layer: 1, pos: 3467
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 2408
type: A, layer: 1, pos: 3289
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 3028
type: A, layer: 1, pos: 3471
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 2551
type: A, layer: 1, pos: 2211
type: A, layer: 1, pos: 3528
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 331
type: A, layer: 1, pos: 2365
type: A, layer: 1, pos: 2213
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 2535
type: A, layer: 1, pos: 2210
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 3016
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 3388
type: A, layer: 1, pos: 3552
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2206
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 3497
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 2341
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 2160
type: A, layer: 1, pos: 2637
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2664
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2508
type: A, layer: 1, pos: 2499
type: A, layer: 1, pos: 3478
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 2938
type: A, layer: 1, pos: 2700
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 3498
type: A, layer: 1, pos: 2187
type: A, layer: 1, pos: 3523
type: A, layer: 1, pos: 3256
type: A, layer: 1, pos: 2937
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 2484
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 2701
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 2936
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 3483
type: A, layer: 1, pos: 312
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 490
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 2487
type: A, layer: 1, pos: 2217
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 372
type: A, layer: 1, pos: 2955
type: A, layer: 1, pos: 2255
type: A, layer: 1, pos: 2942
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 2481
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 3208
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 3298
type: A, layer: 1, pos: 2250
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 3343
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 3313
type: A, layer: 1, pos: 2475
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2609
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3044
type: A, layer: 1, pos: 3164
type: A, layer: 1, pos: 3299
type: A, layer: 1, pos: 3314
type: A, layer: 1, pos: 3509

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2363

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2163823, upper bound: 0.2167224
time: 15.86 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2167199, upper bound: 0.2167233
time: 19.59 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 35.52 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 35.52
Output dim: 5, lower bound: -0.2163823, upper bound: 0.2167224
NS_A2, status: Status.UNKNOWN, split count: 1, time: 35.52
Output dim: 5, lower bound: -0.2167199, upper bound: 0.2167233

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -3.5647852, -1.4761646, -3.5648627, -1.4746096, -1.7909801, 1.7894897
1: -1.7699745, 0.2686715, -1.7700062, 0.2699122, -1.8908833, 1.8896585
2: -1.5417410, -0.9444601, -1.5418679, -0.9444600, -0.1892358, 0.1893852
3: -0.9864948, -0.2778086, -0.9871544, -0.2778010, -0.4557553, 0.4564613
4: -0.7429157, -0.2673028, -0.7430943, -0.2672843, -0.2297600, 0.2298556
5: -0.9673718, -0.1104580, -0.9680575, -0.1104566, -0.4503306, 0.4510857
6: -0.8151966, -0.3731490, -0.8156791, -0.3731488, -0.1587520, 0.1593173
7: -0.7437176, 0.0228641, -0.7439757, 0.0228663, -0.5865616, 0.5868277
8: -3.9530449, -2.1249819, -3.9530697, -2.1246896, -0.9409546, 0.9403028
9: -1.9561429, -0.3185012, -1.9561696, -0.3180630, -1.2866930, 1.2862053

Time for backsubstitution: 5.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 2570
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2388
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 262
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 100
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 2600
type: B, layer: 1, pos: 2331
type: B, layer: 1, pos: 2336
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 2170
type: B, layer: 1, pos: 347
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 287
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 3040
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 403
type: B, layer: 1, pos: 2094
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 3567
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2419
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 2591
type: B, layer: 1, pos: 2553
type: B, layer: 1, pos: 2607
type: B, layer: 1, pos: 317
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 378
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 3459
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 263
type: B, layer: 1, pos: 2132
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 3013
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 3273
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 3112
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 3125
type: B, layer: 1, pos: 387
type: B, layer: 1, pos: 2507
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 2322
type: B, layer: 1, pos: 3467
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 2408
type: B, layer: 1, pos: 3289
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 2510
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 3028
type: B, layer: 1, pos: 3471
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 2211
type: B, layer: 1, pos: 3528
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 331
type: B, layer: 1, pos: 2365
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 2213
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 2535
type: B, layer: 1, pos: 2210
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 3016
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 3388
type: B, layer: 1, pos: 3552
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2206
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 3497
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 2341
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 2160
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2664
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 2508
type: B, layer: 1, pos: 2499
type: B, layer: 1, pos: 3478
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 2938
type: B, layer: 1, pos: 2700
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 3498
type: B, layer: 1, pos: 2187
type: B, layer: 1, pos: 3523
type: B, layer: 1, pos: 3256
type: B, layer: 1, pos: 2937
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 2701
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 2936
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3483
type: B, layer: 1, pos: 312
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 2487
type: B, layer: 1, pos: 2217
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 372
type: B, layer: 1, pos: 2955
type: B, layer: 1, pos: 2255
type: B, layer: 1, pos: 2942
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 2052
type: B, layer: 1, pos: 3208
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 2051
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 3298
type: B, layer: 1, pos: 2250
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 3343
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 3313
type: B, layer: 1, pos: 2475
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2609
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3044
type: B, layer: 1, pos: 3164
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3314
type: B, layer: 1, pos: 3509

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2362

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2161536, upper bound: 0.2163189
time: 17.69 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2161536, upper bound: 0.2166066
time: 130.03 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -3.5730884, -1.4732997, -3.5649979, -1.4724896, -1.8016137, 1.7922970
1: -1.7754083, 0.2718749, -1.7699757, 0.2725458, -1.8992058, 1.8928189
2: -1.5417036, -0.9445955, -1.5417923, -0.9444598, -0.1895701, 0.1901045
3: -0.9864252, -0.2755070, -0.9870252, -0.2777870, -0.4550240, 0.4607754
4: -0.7428526, -0.2670017, -0.7430117, -0.2672453, -0.2293606, 0.2298789
5: -0.9674545, -0.1078289, -0.9680828, -0.1104543, -0.4494556, 0.4565411
6: -0.8162628, -0.3713825, -0.8164965, -0.3731483, -0.1593356, 0.1646165
7: -0.7404180, 0.0226112, -0.7411860, 0.0228694, -0.5837810, 0.5841402
8: -3.9530423, -2.1258750, -3.9531136, -2.1255367, -0.9474283, 0.9411643
9: -1.9579916, -0.3174965, -1.9562230, -0.3172190, -1.2900721, 1.2873628

Time for backsubstitution: 6.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 2364
type: B, layer: 1, pos: 2570
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2388
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 262
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 100
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 2600
type: B, layer: 1, pos: 2331
type: B, layer: 1, pos: 2336
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 2170
type: B, layer: 1, pos: 347
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 287
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 3040
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 403
type: B, layer: 1, pos: 2094
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 3567
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2419
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 2591
type: B, layer: 1, pos: 2553
type: B, layer: 1, pos: 2607
type: B, layer: 1, pos: 317
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 378
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 3459
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 263
type: B, layer: 1, pos: 2132
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 3013
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 3273
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 3112
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 3125
type: B, layer: 1, pos: 387
type: B, layer: 1, pos: 2507
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 2322
type: B, layer: 1, pos: 3467
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 2408
type: B, layer: 1, pos: 3289
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 2510
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 3028
type: B, layer: 1, pos: 3471
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 3528
type: B, layer: 1, pos: 2211
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 331
type: B, layer: 1, pos: 2365
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 2213
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 2535
type: B, layer: 1, pos: 2210
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 3016
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 3388
type: B, layer: 1, pos: 3552
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2206
type: B, layer: 1, pos: 3497
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 2341
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 2160
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2664
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 2508
type: B, layer: 1, pos: 2499
type: B, layer: 1, pos: 3478
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 2938
type: B, layer: 1, pos: 2700
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 3498
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 2187
type: B, layer: 1, pos: 3523
type: B, layer: 1, pos: 3256
type: B, layer: 1, pos: 2937
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 2701
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 2936
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 312
type: B, layer: 1, pos: 3483
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 2487
type: B, layer: 1, pos: 2217
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 372
type: B, layer: 1, pos: 2955
type: B, layer: 1, pos: 2255
type: B, layer: 1, pos: 2942
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 2481
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 2052
type: B, layer: 1, pos: 3208
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 2051
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 3298
type: B, layer: 1, pos: 2250
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 3343
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 3313
type: B, layer: 1, pos: 2475
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2609
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3044
type: B, layer: 1, pos: 3164
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3314
type: B, layer: 1, pos: 3509

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2362

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2166052, upper bound: 0.2163183
time: 198.73 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2166035, upper bound: 0.2166048
time: 140.99 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 345.90 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 345.90
Output dim: 5, lower bound: -0.2161536, upper bound: 0.2163189
NS_A1_B2, status: Status.VERIFIED, split count: 2, time: 345.90
Output dim: 5, lower bound: -0.2161536, upper bound: 0.2166066
NS_A2_B1, status: Status.VERIFIED, split count: 2, time: 345.90
Output dim: 5, lower bound: -0.2166052, upper bound: 0.2163183
NS_A2_B2, status: Status.VERIFIED, split count: 2, time: 345.90
Output dim: 5, lower bound: -0.2166035, upper bound: 0.2166048

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 682.99 + 535.13 = 1218.12 seconds
