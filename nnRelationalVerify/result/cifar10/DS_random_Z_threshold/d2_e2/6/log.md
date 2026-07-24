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
execution time: IAR + RelationalAnalysis = 7.79 + 664.10 = 671.90 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.2168844, upper bound: 0.2168864

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3164
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 3509
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 317
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 485
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 331
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 490
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 3299
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 372
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 2700
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 287
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 792

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2168748, upper bound: 0.2168860
time: 323.11 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2168811, upper bound: 0.2168778
time: 32.29 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 355.41 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 355.41
Output dim: 5, lower bound: -0.2168748, upper bound: 0.2168860
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 355.41
Output dim: 5, lower bound: -0.2168811, upper bound: 0.2168778

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -3.5651972, -1.4679933, -3.5651972, -1.4679933, -1.7979294, 1.7979190
1: -1.7701428, 0.2758193, -1.7701428, 0.2758193, -1.8970163, 1.8970091
2: -1.5424505, -0.9444594, -1.5424505, -0.9444594, -0.1906997, 0.1906987
3: -0.9903108, -0.2777686, -0.9903108, -0.2777686, -0.4601136, 0.4601145
4: -0.7439071, -0.2672053, -0.7439071, -0.2672053, -0.2306994, 0.2306972
5: -0.9715933, -0.1104509, -0.9715933, -0.1104509, -0.4551709, 0.4551766
6: -0.8177381, -0.3731477, -0.8177381, -0.3731477, -0.1619202, 0.1619202
7: -0.7450799, 0.0228758, -0.7450799, 0.0228758, -0.5879828, 0.5879833
8: -3.9531717, -2.1230025, -3.9531717, -2.1230025, -0.9437667, 0.9437129
9: -1.9562843, -0.3158286, -1.9562843, -0.3158286, -1.2890590, 1.2890327

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 287
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 3164
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 317
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2700
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 490
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3509
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 485
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 3299
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 331
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 372
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 684

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 675

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2168742, upper bound: 0.2168871
time: 24.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2168744, upper bound: 0.2168838
time: 74.18 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -3.5651972, -1.4679933, -3.5651972, -1.4679933, -1.7979189, 1.7979295
1: -1.7701428, 0.2758193, -1.7701428, 0.2758193, -1.8970087, 1.8970168
2: -1.5424505, -0.9444594, -1.5424505, -0.9444594, -0.1906987, 0.1906997
3: -0.9903108, -0.2777686, -0.9903108, -0.2777686, -0.4601145, 0.4601136
4: -0.7439071, -0.2672053, -0.7439071, -0.2672053, -0.2306971, 0.2306994
5: -0.9715933, -0.1104509, -0.9715933, -0.1104509, -0.4551766, 0.4551709
6: -0.8177381, -0.3731477, -0.8177381, -0.3731477, -0.1619202, 0.1619202
7: -0.7450799, 0.0228758, -0.7450799, 0.0228758, -0.5879833, 0.5879828
8: -3.9531717, -2.1230025, -3.9531717, -2.1230025, -0.9437129, 0.9437666
9: -1.9562843, -0.3158286, -1.9562843, -0.3158286, -1.2890328, 1.2890589

Time for backsubstitution: 6.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 3164
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 287
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 2700
type: DSZ, layer: 1, pos: 485
type: DSZ, layer: 1, pos: 331
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 490
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 3299
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 3509
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 317
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 372
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2160

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 262

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2168815, upper bound: 0.2161826
time: 237.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2161899, upper bound: 0.2168765
time: 40.20 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 283.54 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 283.54
Output dim: 5, lower bound: -0.2168742, upper bound: 0.2168871
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 283.54
Output dim: 5, lower bound: -0.2168744, upper bound: 0.2168838
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 283.54
Output dim: 5, lower bound: -0.2168815, upper bound: 0.2161826
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 283.54
Output dim: 5, lower bound: -0.2161899, upper bound: 0.2168765

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.5651972, -1.4679933, -3.5651972, -1.4679933, -1.7979127, 1.7978876
1: -1.7701428, 0.2758193, -1.7701428, 0.2758193, -1.8970160, 1.8970087
2: -1.5424505, -0.9444594, -1.5424505, -0.9444594, -0.1906977, 0.1906953
3: -0.9903108, -0.2777686, -0.9903108, -0.2777686, -0.4601136, 0.4601144
4: -0.7439071, -0.2672053, -0.7439071, -0.2672053, -0.2306963, 0.2306913
5: -0.9715933, -0.1104509, -0.9715933, -0.1104509, -0.4551701, 0.4551761
6: -0.8177381, -0.3731477, -0.8177381, -0.3731477, -0.1619198, 0.1619201
7: -0.7450799, 0.0228758, -0.7450799, 0.0228758, -0.5879828, 0.5879833
8: -3.9531717, -2.1230025, -3.9531717, -2.1230025, -0.9437591, 0.9436936
9: -1.9562843, -0.3158286, -1.9562843, -0.3158286, -1.2890587, 1.2890325

Time for backsubstitution: 6.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 490
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 331
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2700
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 287
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 485
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 317
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 3509
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 3299
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 3164
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 372
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 2145

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2597

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2168134, upper bound: 0.2167268
time: 442.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2167162, upper bound: 0.2168232
time: 52.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.5651972, -1.4679933, -3.5651972, -1.4679933, -1.7978982, 1.7979021
1: -1.7701428, 0.2758193, -1.7701428, 0.2758193, -1.8970158, 1.8970087
2: -1.5424505, -0.9444594, -1.5424505, -0.9444594, -0.1906963, 0.1906967
3: -0.9903108, -0.2777686, -0.9903108, -0.2777686, -0.4601136, 0.4601144
4: -0.7439071, -0.2672053, -0.7439071, -0.2672053, -0.2306935, 0.2306941
5: -0.9715933, -0.1104509, -0.9715933, -0.1104509, -0.4551705, 0.4551758
6: -0.8177381, -0.3731477, -0.8177381, -0.3731477, -0.1619201, 0.1619198
7: -0.7450799, 0.0228758, -0.7450799, 0.0228758, -0.5879829, 0.5879832
8: -3.9531717, -2.1230025, -3.9531717, -2.1230025, -0.9437475, 0.9437051
9: -1.9562843, -0.3158286, -1.9562843, -0.3158286, -1.2890587, 1.2890327

Time for backsubstitution: 5.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2700
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 287
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 317
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 3509
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 3164
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 485
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 331
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 490
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 3299
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 372
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2052

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2700

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2168683, upper bound: 0.2168853
time: 333.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2168744, upper bound: 0.2168781
time: 143.43 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.5651972, -1.4679933, -3.5651972, -1.4679933, -1.8015879, 1.8018204
1: -1.7701428, 0.2758193, -1.7701428, 0.2758193, -1.8962723, 1.8962407
2: -1.5424505, -0.9444594, -1.5424505, -0.9444594, -0.1889596, 0.1888671
3: -0.9903108, -0.2777686, -0.9903108, -0.2777686, -0.4589963, 0.4587419
4: -0.7439071, -0.2672053, -0.7439071, -0.2672053, -0.2184936, 0.2191137
5: -0.9715933, -0.1104509, -0.9715933, -0.1104509, -0.4439599, 0.4432816
6: -0.8177381, -0.3731477, -0.8177381, -0.3731477, -0.1588817, 0.1590412
7: -0.7450799, 0.0228758, -0.7450799, 0.0228758, -0.5882674, 0.5882745
8: -3.9531717, -2.1230025, -3.9531717, -2.1230025, -0.9350498, 0.9355490
9: -1.9562843, -0.3158286, -1.9562843, -0.3158286, -1.2867256, 1.2866271

Time for backsubstitution: 5.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 331
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 287
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2700
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 485
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 3164
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 3299
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 3509
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 490
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 372
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 317
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2168514, upper bound: 0.2161649
time: 7.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2168616, upper bound: 0.2161541
time: 14.94 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.5651972, -1.4679933, -3.5651972, -1.4679933, -1.8018101, 1.8015984
1: -1.7701428, 0.2758193, -1.7701428, 0.2758193, -1.8962327, 1.8962803
2: -1.5424505, -0.9444594, -1.5424505, -0.9444594, -0.1888661, 0.1889605
3: -0.9903108, -0.2777686, -0.9903108, -0.2777686, -0.4587427, 0.4589954
4: -0.7439071, -0.2672053, -0.7439071, -0.2672053, -0.2191114, 0.2184959
5: -0.9715933, -0.1104509, -0.9715933, -0.1104509, -0.4432873, 0.4439541
6: -0.8177381, -0.3731477, -0.8177381, -0.3731477, -0.1590412, 0.1588818
7: -0.7450799, 0.0228758, -0.7450799, 0.0228758, -0.5882749, 0.5882670
8: -3.9531717, -2.1230025, -3.9531717, -2.1230025, -0.9354954, 0.9351034
9: -1.9562843, -0.3158286, -1.9562843, -0.3158286, -1.2866009, 1.2867520

Time for backsubstitution: 5.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3509
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 3299
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 287
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 317
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 485
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 490
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 331
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 3164
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2700
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 372
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 741

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2200

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2160619, upper bound: 0.2167506
time: 36.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2160619, upper bound: 0.2167489
time: 151.50 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 193.52 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 193.52
Output dim: 5, lower bound: -0.2168134, upper bound: 0.2167268
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 193.52
Output dim: 5, lower bound: -0.2167162, upper bound: 0.2168232
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 193.52
Output dim: 5, lower bound: -0.2168683, upper bound: 0.2168853
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 193.52
Output dim: 5, lower bound: -0.2168744, upper bound: 0.2168781
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 193.52
Output dim: 5, lower bound: -0.2168514, upper bound: 0.2161649
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 193.52
Output dim: 5, lower bound: -0.2168616, upper bound: 0.2161541
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 193.52
Output dim: 5, lower bound: -0.2160619, upper bound: 0.2167506
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 193.52
Output dim: 5, lower bound: -0.2160619, upper bound: 0.2167489

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.5651972, -1.4679933, -3.5651972, -1.4679933, -1.7964290, 1.7965333
1: -1.7701428, 0.2758193, -1.7701428, 0.2758193, -1.8959956, 1.8960557
2: -1.5424505, -0.9444594, -1.5424505, -0.9444594, -0.1903560, 0.1903414
3: -0.9903108, -0.2777686, -0.9903108, -0.2777686, -0.4588550, 0.4587796
4: -0.7439071, -0.2672053, -0.7439071, -0.2672053, -0.2304979, 0.2304839
5: -0.9715933, -0.1104509, -0.9715933, -0.1104509, -0.4532189, 0.4531325
6: -0.8177381, -0.3731477, -0.8177381, -0.3731477, -0.1617658, 0.1617603
7: -0.7450799, 0.0228758, -0.7450799, 0.0228758, -0.5872864, 0.5872560
8: -3.9531717, -2.1230025, -3.9531717, -2.1230025, -0.9356675, 0.9359905
9: -1.9562843, -0.3158286, -1.9562843, -0.3158286, -1.2843722, 1.2845228

Time for backsubstitution: 5.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 331
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2700
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 3299
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 485
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 372
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3164
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 490
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 287
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 317
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 3509
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 3467

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2701

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2168104, upper bound: 0.2167238
time: 189.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2168115, upper bound: 0.2167212
time: 42.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.5651972, -1.4679933, -3.5651972, -1.4679933, -1.7965581, 1.7964041
1: -1.7701428, 0.2758193, -1.7701428, 0.2758193, -1.8960638, 1.8959875
2: -1.5424505, -0.9444594, -1.5424505, -0.9444594, -0.1903438, 0.1903537
3: -0.9903108, -0.2777686, -0.9903108, -0.2777686, -0.4587787, 0.4588559
4: -0.7439071, -0.2672053, -0.7439071, -0.2672053, -0.2304889, 0.2304929
5: -0.9715933, -0.1104509, -0.9715933, -0.1104509, -0.4531265, 0.4532249
6: -0.8177381, -0.3731477, -0.8177381, -0.3731477, -0.1617601, 0.1617660
7: -0.7450799, 0.0228758, -0.7450799, 0.0228758, -0.5872554, 0.5872869
8: -3.9531717, -2.1230025, -3.9531717, -2.1230025, -0.9360558, 0.9356021
9: -1.9562843, -0.3158286, -1.9562843, -0.3158286, -1.2845489, 1.2843461

Time for backsubstitution: 5.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 485
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 331
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 317
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 287
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2700
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 490
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 372
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 3164
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3299
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 3509
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 2193

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 689

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2167160, upper bound: 0.2167248
time: 424.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2167160, upper bound: 0.2168226
time: 379.78 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 810.01 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 810.01
Output dim: 5, lower bound: -0.2168104, upper bound: 0.2167238
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 810.01
Output dim: 5, lower bound: -0.2168115, upper bound: 0.2167212
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 810.01
Output dim: 5, lower bound: -0.2167160, upper bound: 0.2167248
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 810.01
Output dim: 5, lower bound: -0.2167160, upper bound: 0.2168226
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 810.01
Output dim: 5, lower bound: -0.2168683, upper bound: 0.2168853
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 810.01
Output dim: 5, lower bound: -0.2168744, upper bound: 0.2168781
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 810.01
Output dim: 5, lower bound: -0.2168514, upper bound: 0.2161649
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 810.01
Output dim: 5, lower bound: -0.2168616, upper bound: 0.2161541
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 810.01
Output dim: 5, lower bound: -0.2160619, upper bound: 0.2167506
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 810.01
Output dim: 5, lower bound: -0.2160619, upper bound: 0.2167489

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 671.90 + 2997.82 = 3669.72 seconds
