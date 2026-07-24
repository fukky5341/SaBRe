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
execution time: IAR + RelationalAnalysis = 8.10 + 661.48 = 669.59 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.2168844, upper bound: 0.2168864

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 485
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 331
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 372
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 287
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 317
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 490
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 2700
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 3164
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 3299
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 3509
type: DSZ, layer: 1, pos: 3567

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2170

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2166683, upper bound: 0.2168359
time: 36.68 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2168350, upper bound: 0.2166699
time: 112.20 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 148.96 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 148.96
Output dim: 5, lower bound: -0.2166683, upper bound: 0.2168359
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 148.96
Output dim: 5, lower bound: -0.2168350, upper bound: 0.2166699

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -3.5651972, -1.4679933, -3.5651972, -1.4679933, -1.7976747, 1.7976532
1: -1.7701428, 0.2758193, -1.7701428, 0.2758193, -1.8969004, 1.8968890
2: -1.5424505, -0.9444594, -1.5424505, -0.9444594, -0.1905252, 0.1905243
3: -0.9903108, -0.2777686, -0.9903108, -0.2777686, -0.4596296, 0.4596619
4: -0.7439071, -0.2672053, -0.7439071, -0.2672053, -0.2306428, 0.2306422
5: -0.9715933, -0.1104509, -0.9715933, -0.1104509, -0.4546072, 0.4546459
6: -0.8177381, -0.3731477, -0.8177381, -0.3731477, -0.1617783, 0.1617871
7: -0.7450799, 0.0228758, -0.7450799, 0.0228758, -0.5878061, 0.5878052
8: -3.9531717, -2.1230025, -3.9531717, -2.1230025, -0.9433758, 0.9433432
9: -1.9562843, -0.3158286, -1.9562843, -0.3158286, -1.2889808, 1.2889676

Time for backsubstitution: 6.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 485
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 331
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 372
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 287
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 317
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 490
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 2700
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 3164
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 3299
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 3509
type: DSZ, layer: 1, pos: 3567

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2171

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2165630, upper bound: 0.2165639
time: 171.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2165994, upper bound: 0.2166748
time: 113.76 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -3.5651972, -1.4679933, -3.5651972, -1.4679933, -1.7976533, 1.7976744
1: -1.7701428, 0.2758193, -1.7701428, 0.2758193, -1.8968892, 1.8969004
2: -1.5424505, -0.9444594, -1.5424505, -0.9444594, -0.1905243, 0.1905252
3: -0.9903108, -0.2777686, -0.9903108, -0.2777686, -0.4596619, 0.4596296
4: -0.7439071, -0.2672053, -0.7439071, -0.2672053, -0.2306422, 0.2306428
5: -0.9715933, -0.1104509, -0.9715933, -0.1104509, -0.4546459, 0.4546072
6: -0.8177381, -0.3731477, -0.8177381, -0.3731477, -0.1617871, 0.1617783
7: -0.7450799, 0.0228758, -0.7450799, 0.0228758, -0.5878051, 0.5878061
8: -3.9531717, -2.1230025, -3.9531717, -2.1230025, -0.9433433, 0.9433761
9: -1.9562843, -0.3158286, -1.9562843, -0.3158286, -1.2889675, 1.2889812

Time for backsubstitution: 6.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 485
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 331
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 372
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 287
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 317
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 490
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 2700
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 3164
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 3299
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 3509
type: DSZ, layer: 1, pos: 3567

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2171

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2166758, upper bound: 0.2166035
time: 122.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2168079, upper bound: 0.2165652
time: 23.73 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 152.53 seconds
DS_DSZ1_DSZ1, status: Status.VERIFIED, split count: 2, time: 152.53
Output dim: 5, lower bound: -0.2165630, upper bound: 0.2165639
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 152.53
Output dim: 5, lower bound: -0.2165994, upper bound: 0.2166748
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 152.53
Output dim: 5, lower bound: -0.2166758, upper bound: 0.2166035
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 152.53
Output dim: 5, lower bound: -0.2168079, upper bound: 0.2165652

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.5651972, -1.4679933, -3.5651972, -1.4679933, -1.7971579, 1.7971448
1: -1.7701428, 0.2758193, -1.7701428, 0.2758193, -1.8966563, 1.8966465
2: -1.5424505, -0.9444594, -1.5424505, -0.9444594, -0.1902282, 0.1902291
3: -0.9903108, -0.2777686, -0.9903108, -0.2777686, -0.4592372, 0.4592469
4: -0.7439071, -0.2672053, -0.7439071, -0.2672053, -0.2305319, 0.2305318
5: -0.9715933, -0.1104509, -0.9715933, -0.1104509, -0.4540781, 0.4540917
6: -0.8177381, -0.3731477, -0.8177381, -0.3731477, -0.1615664, 0.1615741
7: -0.7450799, 0.0228758, -0.7450799, 0.0228758, -0.5875440, 0.5875458
8: -3.9531717, -2.1230025, -3.9531717, -2.1230025, -0.9421905, 0.9421428
9: -1.9562843, -0.3158286, -1.9562843, -0.3158286, -1.2886177, 1.2886047

Time for backsubstitution: 6.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 485
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 331
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 372
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 287
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 317
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 490
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 2700
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 3164
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 3299
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 3509
type: DSZ, layer: 1, pos: 3567

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2607

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2165658, upper bound: 0.2165779
time: 33.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2165169, upper bound: 0.2166530
time: 8.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.5651972, -1.4679933, -3.5651972, -1.4679933, -1.7971448, 1.7971581
1: -1.7701428, 0.2758193, -1.7701428, 0.2758193, -1.8966460, 1.8966565
2: -1.5424505, -0.9444594, -1.5424505, -0.9444594, -0.1902291, 0.1902282
3: -0.9903108, -0.2777686, -0.9903108, -0.2777686, -0.4592468, 0.4592372
4: -0.7439071, -0.2672053, -0.7439071, -0.2672053, -0.2305318, 0.2305319
5: -0.9715933, -0.1104509, -0.9715933, -0.1104509, -0.4540917, 0.4540781
6: -0.8177381, -0.3731477, -0.8177381, -0.3731477, -0.1615741, 0.1615664
7: -0.7450799, 0.0228758, -0.7450799, 0.0228758, -0.5875459, 0.5875439
8: -3.9531717, -2.1230025, -3.9531717, -2.1230025, -0.9421426, 0.9421905
9: -1.9562843, -0.3158286, -1.9562843, -0.3158286, -1.2886047, 1.2886176

Time for backsubstitution: 6.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 485
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 331
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 372
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 287
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 317
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 490
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 2700
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 3164
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 3299
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 3509
type: DSZ, layer: 1, pos: 3567

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 2607

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2166475, upper bound: 0.2165197
time: 476.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2165778, upper bound: 0.2165679
time: 45.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.5651972, -1.4679933, -3.5651972, -1.4679933, -1.7971364, 1.7971767
1: -1.7701428, 0.2758193, -1.7701428, 0.2758193, -1.8966451, 1.8966603
2: -1.5424505, -0.9444594, -1.5424505, -0.9444594, -0.1902292, 0.1902300
3: -0.9903108, -0.2777686, -0.9903108, -0.2777686, -0.4592831, 0.4592146
4: -0.7439071, -0.2672053, -0.7439071, -0.2672053, -0.2305318, 0.2305323
5: -0.9715933, -0.1104509, -0.9715933, -0.1104509, -0.4541382, 0.4540530
6: -0.8177381, -0.3731477, -0.8177381, -0.3731477, -0.1615801, 0.1615653
7: -0.7450799, 0.0228758, -0.7450799, 0.0228758, -0.5875481, 0.5875468
8: -3.9531717, -2.1230025, -3.9531717, -2.1230025, -0.9421581, 0.9421929
9: -1.9562843, -0.3158286, -1.9562843, -0.3158286, -1.2886044, 1.2886229

Time for backsubstitution: 6.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 485
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 331
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 372
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 287
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 317
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 490
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 2700
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 3164
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 3299
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 3509
type: DSZ, layer: 1, pos: 3567

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2607

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2167712, upper bound: 0.2164816
time: 54.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2167228, upper bound: 0.2165308
time: 177.33 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 238.73 seconds
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 238.73
Output dim: 5, lower bound: -0.2165658, upper bound: 0.2165779
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 238.73
Output dim: 5, lower bound: -0.2165169, upper bound: 0.2166530
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 238.73
Output dim: 5, lower bound: -0.2166475, upper bound: 0.2165197
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 238.73
Output dim: 5, lower bound: -0.2165778, upper bound: 0.2165679
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 238.73
Output dim: 5, lower bound: -0.2167712, upper bound: 0.2164816
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 238.73
Output dim: 5, lower bound: -0.2167228, upper bound: 0.2165308

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.5651972, -1.4679933, -3.5651972, -1.4679933, -1.7971298, 1.7971697
1: -1.7701428, 0.2758193, -1.7701428, 0.2758193, -1.8966484, 1.8966634
2: -1.5424505, -0.9444594, -1.5424505, -0.9444594, -0.1902414, 0.1902422
3: -0.9903108, -0.2777686, -0.9903108, -0.2777686, -0.4592830, 0.4592144
4: -0.7439071, -0.2672053, -0.7439071, -0.2672053, -0.2305489, 0.2305499
5: -0.9715933, -0.1104509, -0.9715933, -0.1104509, -0.4541402, 0.4540549
6: -0.8177381, -0.3731477, -0.8177381, -0.3731477, -0.1615850, 0.1615704
7: -0.7450799, 0.0228758, -0.7450799, 0.0228758, -0.5875559, 0.5875548
8: -3.9531717, -2.1230025, -3.9531717, -2.1230025, -0.9421582, 0.9421930
9: -1.9562843, -0.3158286, -1.9562843, -0.3158286, -1.2886051, 1.2886243

Time for backsubstitution: 6.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 485
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 331
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 372
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 287
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 317
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 490
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 2700
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 3164
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 3299
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 3509
type: DSZ, layer: 1, pos: 3567

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 130

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2163748, upper bound: 0.2164523
time: 33.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2167654, upper bound: 0.2161925
time: 634.86 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.5651972, -1.4679933, -3.5651972, -1.4679933, -1.7971299, 1.7971697
1: -1.7701428, 0.2758193, -1.7701428, 0.2758193, -1.8966484, 1.8966634
2: -1.5424505, -0.9444594, -1.5424505, -0.9444594, -0.1902414, 0.1902422
3: -0.9903108, -0.2777686, -0.9903108, -0.2777686, -0.4592830, 0.4592146
4: -0.7439071, -0.2672053, -0.7439071, -0.2672053, -0.2305494, 0.2305493
5: -0.9715933, -0.1104509, -0.9715933, -0.1104509, -0.4541400, 0.4540552
6: -0.8177381, -0.3731477, -0.8177381, -0.3731477, -0.1615852, 0.1615702
7: -0.7450799, 0.0228758, -0.7450799, 0.0228758, -0.5875560, 0.5875546
8: -3.9531717, -2.1230025, -3.9531717, -2.1230025, -0.9421583, 0.9421930
9: -1.9562843, -0.3158286, -1.9562843, -0.3158286, -1.2886052, 1.2886243

Time for backsubstitution: 6.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 485
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 331
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 372
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 287
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 317
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 490
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 2700
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 3164
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 3299
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 3509
type: DSZ, layer: 1, pos: 3567

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 130

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2163268, upper bound: 0.2165039
time: 25.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2167173, upper bound: 0.2162453
time: 13.58 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 45.93 seconds
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 45.93
Output dim: 5, lower bound: -0.2163748, upper bound: 0.2164523
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 45.93
Output dim: 5, lower bound: -0.2167654, upper bound: 0.2161925
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 45.93
Output dim: 5, lower bound: -0.2163268, upper bound: 0.2165039
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 45.93
Output dim: 5, lower bound: -0.2167173, upper bound: 0.2162453

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.5651972, -1.4679933, -3.5651972, -1.4679933, -1.7959367, 1.7960378
1: -1.7701428, 0.2758193, -1.7701428, 0.2758193, -1.8961025, 1.8961444
2: -1.5424505, -0.9444594, -1.5424505, -0.9444594, -0.1898523, 0.1898283
3: -0.9903108, -0.2777686, -0.9903108, -0.2777686, -0.4580040, 0.4578666
4: -0.7439071, -0.2672053, -0.7439071, -0.2672053, -0.2303525, 0.2303452
5: -0.9715933, -0.1104509, -0.9715933, -0.1104509, -0.4524930, 0.4523194
6: -0.8177381, -0.3731477, -0.8177381, -0.3731477, -0.1612562, 0.1612252
7: -0.7450799, 0.0228758, -0.7450799, 0.0228758, -0.5870043, 0.5869750
8: -3.9531717, -2.1230025, -3.9531717, -2.1230025, -0.9396247, 0.9397879
9: -1.9562843, -0.3158286, -1.9562843, -0.3158286, -1.2878382, 1.2878959

Time for backsubstitution: 6.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 485
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 331
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 372
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 287
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 317
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 490
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 2700
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 3164
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 3299
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 3509
type: DSZ, layer: 1, pos: 3567

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2408

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2166880, upper bound: 0.2161200
time: 139.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2166923, upper bound: 0.2161168
time: 62.10 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.5651972, -1.4679933, -3.5651972, -1.4679933, -1.7959368, 1.7960378
1: -1.7701428, 0.2758193, -1.7701428, 0.2758193, -1.8961025, 1.8961444
2: -1.5424505, -0.9444594, -1.5424505, -0.9444594, -0.1898524, 0.1898283
3: -0.9903108, -0.2777686, -0.9903108, -0.2777686, -0.4580039, 0.4578667
4: -0.7439071, -0.2672053, -0.7439071, -0.2672053, -0.2303531, 0.2303447
5: -0.9715933, -0.1104509, -0.9715933, -0.1104509, -0.4524927, 0.4523196
6: -0.8177381, -0.3731477, -0.8177381, -0.3731477, -0.1612563, 0.1612250
7: -0.7450799, 0.0228758, -0.7450799, 0.0228758, -0.5870045, 0.5869749
8: -3.9531717, -2.1230025, -3.9531717, -2.1230025, -0.9396248, 0.9397879
9: -1.9562843, -0.3158286, -1.9562843, -0.3158286, -1.2878382, 1.2878959

Time for backsubstitution: 6.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 485
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 331
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 372
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 287
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 317
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 490
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 2700
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 3164
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 3299
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 3509
type: DSZ, layer: 1, pos: 3567

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2408

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2166420, upper bound: 0.2161671
time: 185.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2166441, upper bound: 0.2161675
time: 11.95 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 203.66 seconds
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 203.66
Output dim: 5, lower bound: -0.2166880, upper bound: 0.2161200
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 203.66
Output dim: 5, lower bound: -0.2166923, upper bound: 0.2161168
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 203.66
Output dim: 5, lower bound: -0.2166420, upper bound: 0.2161671
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 203.66
Output dim: 5, lower bound: -0.2166441, upper bound: 0.2161675

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.5651972, -1.4679933, -3.5651972, -1.4679933, -1.7957948, 1.7958738
1: -1.7701428, 0.2758193, -1.7701428, 0.2758193, -1.8960311, 1.8960557
2: -1.5424505, -0.9444594, -1.5424505, -0.9444594, -0.1898507, 0.1898284
3: -0.9903108, -0.2777686, -0.9903108, -0.2777686, -0.4579767, 0.4578367
4: -0.7439071, -0.2672053, -0.7439071, -0.2672053, -0.2303458, 0.2303407
5: -0.9715933, -0.1104509, -0.9715933, -0.1104509, -0.4524454, 0.4522772
6: -0.8177381, -0.3731477, -0.8177381, -0.3731477, -0.1612549, 0.1612239
7: -0.7450799, 0.0228758, -0.7450799, 0.0228758, -0.5869874, 0.5869610
8: -3.9531717, -2.1230025, -3.9531717, -2.1230025, -0.9391671, 0.9392035
9: -1.9562843, -0.3158286, -1.9562843, -0.3158286, -1.2876163, 1.2876093

Time for backsubstitution: 6.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3528
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 387
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 485
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 331
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 3273
type: DSZ, layer: 1, pos: 372
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 347
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 312
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 287
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 317
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 490
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 601
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2094
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 2700
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2969
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 3164
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 3299
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3471
type: DSZ, layer: 1, pos: 3509
type: DSZ, layer: 1, pos: 3567

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2200

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2165474, upper bound: 0.2159800
time: 362.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2165468, upper bound: 0.2159813
time: 26.04 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 394.99 seconds
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 394.99
Output dim: 5, lower bound: -0.2165474, upper bound: 0.2159800
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 394.99
Output dim: 5, lower bound: -0.2165468, upper bound: 0.2159813
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 394.99
Output dim: 5, lower bound: -0.2166923, upper bound: 0.2161168

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 669.59 + 2936.56 = 3606.14 seconds
