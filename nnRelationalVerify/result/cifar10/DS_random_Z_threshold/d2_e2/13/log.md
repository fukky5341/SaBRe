## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 13)
Time budget: 3600 seconds
Split limit: 100
Threshold: 0.0929141992


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3.2739444, -1.3582740, -3.2739444, -1.3582740, -1.5935667, 1.5935667)
1: (-3.1609457, -1.3756382, -3.1609457, -1.3756382, -1.2123365, 1.2123365)
2: (-1.0850873, -0.6183419, -1.0850873, -0.6183419, -0.2405109, 0.2405109)
3: (0.3414853, 0.6887888, 0.3414853, 0.6887888, -0.2271691, 0.2271691)
4: (-4.2210059, -3.3748572, -4.2210059, -3.3748572, -0.2738218, 0.2738218)
5: (1.8216184, 2.1860926, 1.8216184, 2.1860926, -0.1308615, 0.1308615)
6: (-4.4172969, -3.4960172, -4.4172969, -3.4960172, -0.1781177, 0.1781177)
7: (-0.4386549, 0.0872523, -0.4386549, 0.0872523, -0.1360928, 0.1360928)
8: (-3.6494575, -1.5500991, -3.6494575, -1.5500991, -0.9182687, 0.9182688)
9: (-2.9068365, -1.1850188, -2.9068365, -1.1850188, -0.7928562, 0.7928562)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.32 + 77.13 = 84.45 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0930953, upper bound: 0.0931023

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 651
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 3525
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 361
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2317

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2219

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930948, upper bound: 0.0930997
time: 43.23 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930948, upper bound: 0.0931025
time: 13.73 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 56.97 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 56.97
Output dim: 5, lower bound: -0.0930948, upper bound: 0.0930997
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 56.97
Output dim: 5, lower bound: -0.0930948, upper bound: 0.0931025

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -3.2739444, -1.3582740, -3.2739444, -1.3582740, -1.5935667, 1.5935667
1: -3.1609457, -1.3756382, -3.1609457, -1.3756382, -1.2123365, 1.2123365
2: -1.0850873, -0.6183419, -1.0850873, -0.6183419, -0.2405109, 0.2405109
3: 0.3414853, 0.6887888, 0.3414853, 0.6887888, -0.2271691, 0.2271691
4: -4.2210059, -3.3748572, -4.2210059, -3.3748572, -0.2738218, 0.2738218
5: 1.8216184, 2.1860926, 1.8216184, 2.1860926, -0.1308615, 0.1308615
6: -4.4172969, -3.4960172, -4.4172969, -3.4960172, -0.1781177, 0.1781177
7: -0.4386549, 0.0872523, -0.4386549, 0.0872523, -0.1360928, 0.1360928
8: -3.6494575, -1.5500991, -3.6494575, -1.5500991, -0.9182687, 0.9182688
9: -2.9068365, -1.1850188, -2.9068365, -1.1850188, -0.7928562, 0.7928562

Time for backsubstitution: 5.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 3525
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 361
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 651
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 558

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 767

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930857, upper bound: 0.0930985
time: 134.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930960, upper bound: 0.0930908
time: 191.70 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -3.2739444, -1.3582740, -3.2739444, -1.3582740, -1.5935667, 1.5935667
1: -3.1609457, -1.3756382, -3.1609457, -1.3756382, -1.2123365, 1.2123365
2: -1.0850873, -0.6183419, -1.0850873, -0.6183419, -0.2405109, 0.2405109
3: 0.3414853, 0.6887888, 0.3414853, 0.6887888, -0.2271691, 0.2271691
4: -4.2210059, -3.3748572, -4.2210059, -3.3748572, -0.2738218, 0.2738218
5: 1.8216184, 2.1860926, 1.8216184, 2.1860926, -0.1308615, 0.1308615
6: -4.4172969, -3.4960172, -4.4172969, -3.4960172, -0.1781177, 0.1781177
7: -0.4386549, 0.0872523, -0.4386549, 0.0872523, -0.1360928, 0.1360928
8: -3.6494575, -1.5500991, -3.6494575, -1.5500991, -0.9182687, 0.9182688
9: -2.9068365, -1.1850188, -2.9068365, -1.1850188, -0.7928562, 0.7928562

Time for backsubstitution: 5.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 3525
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 651
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 361
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 2451

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2450

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930740, upper bound: 0.0930790
time: 152.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930740, upper bound: 0.0930770
time: 164.02 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 322.22 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 322.22
Output dim: 5, lower bound: -0.0930857, upper bound: 0.0930985
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 322.22
Output dim: 5, lower bound: -0.0930960, upper bound: 0.0930908
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 322.22
Output dim: 5, lower bound: -0.0930740, upper bound: 0.0930790
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 322.22
Output dim: 5, lower bound: -0.0930740, upper bound: 0.0930770

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.2739444, -1.3582740, -3.2739444, -1.3582740, -1.5926598, 1.5926192
1: -3.1609457, -1.3756382, -3.1609457, -1.3756382, -1.2115527, 1.2115600
2: -1.0850873, -0.6183419, -1.0850873, -0.6183419, -0.2404670, 0.2404646
3: 0.3414853, 0.6887888, 0.3414853, 0.6887888, -0.2271309, 0.2271322
4: -4.2210059, -3.3748572, -4.2210059, -3.3748572, -0.2736208, 0.2736109
5: 1.8216184, 2.1860926, 1.8216184, 2.1860926, -0.1307593, 0.1307635
6: -4.4172969, -3.4960172, -4.4172969, -3.4960172, -0.1778649, 0.1778751
7: -0.4386549, 0.0872523, -0.4386549, 0.0872523, -0.1359916, 0.1359958
8: -3.6494575, -1.5500991, -3.6494575, -1.5500991, -0.9158252, 0.9157161
9: -2.9068365, -1.1850188, -2.9068365, -1.1850188, -0.7912588, 0.7912657

Time for backsubstitution: 5.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 3525
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 361
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 651
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 708

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2427

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930521, upper bound: 0.0930721
time: 8.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930521, upper bound: 0.0930721
time: 8.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.2739444, -1.3582740, -3.2739444, -1.3582740, -1.5926191, 1.5926600
1: -3.1609457, -1.3756382, -3.1609457, -1.3756382, -1.2115601, 1.2115526
2: -1.0850873, -0.6183419, -1.0850873, -0.6183419, -0.2404646, 0.2404670
3: 0.3414853, 0.6887888, 0.3414853, 0.6887888, -0.2271322, 0.2271309
4: -4.2210059, -3.3748572, -4.2210059, -3.3748572, -0.2736109, 0.2736208
5: 1.8216184, 2.1860926, 1.8216184, 2.1860926, -0.1307635, 0.1307593
6: -4.4172969, -3.4960172, -4.4172969, -3.4960172, -0.1778751, 0.1778650
7: -0.4386549, 0.0872523, -0.4386549, 0.0872523, -0.1359958, 0.1359916
8: -3.6494575, -1.5500991, -3.6494575, -1.5500991, -0.9157161, 0.9158252
9: -2.9068365, -1.1850188, -2.9068365, -1.1850188, -0.7912656, 0.7912588

Time for backsubstitution: 5.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 651
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 3525
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 361
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 3125

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2926

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930942, upper bound: 0.0930864
time: 190.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930943, upper bound: 0.0930861
time: 119.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.2739444, -1.3582740, -3.2739444, -1.3582740, -1.5933979, 1.5933678
1: -3.1609457, -1.3756382, -3.1609457, -1.3756382, -1.2115188, 1.2114534
2: -1.0850873, -0.6183419, -1.0850873, -0.6183419, -0.2405020, 0.2405038
3: 0.3414853, 0.6887888, 0.3414853, 0.6887888, -0.2271676, 0.2271675
4: -4.2210059, -3.3748572, -4.2210059, -3.3748572, -0.2733406, 0.2734104
5: 1.8216184, 2.1860926, 1.8216184, 2.1860926, -0.1308306, 0.1308315
6: -4.4172969, -3.4960172, -4.4172969, -3.4960172, -0.1781175, 0.1781174
7: -0.4386549, 0.0872523, -0.4386549, 0.0872523, -0.1360439, 0.1360535
8: -3.6494575, -1.5500991, -3.6494575, -1.5500991, -0.9170370, 0.9167327
9: -2.9068365, -1.1850188, -2.9068365, -1.1850188, -0.7914891, 0.7912374

Time for backsubstitution: 6.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 651
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 3525
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 361
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 1110

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 198

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930531, upper bound: 0.0930804
time: 17.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930723, upper bound: 0.0930584
time: 242.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.2739444, -1.3582740, -3.2739444, -1.3582740, -1.5933678, 1.5933979
1: -3.1609457, -1.3756382, -3.1609457, -1.3756382, -1.2114532, 1.2115188
2: -1.0850873, -0.6183419, -1.0850873, -0.6183419, -0.2405038, 0.2405020
3: 0.3414853, 0.6887888, 0.3414853, 0.6887888, -0.2271675, 0.2271676
4: -4.2210059, -3.3748572, -4.2210059, -3.3748572, -0.2734104, 0.2733406
5: 1.8216184, 2.1860926, 1.8216184, 2.1860926, -0.1308315, 0.1308306
6: -4.4172969, -3.4960172, -4.4172969, -3.4960172, -0.1781174, 0.1781175
7: -0.4386549, 0.0872523, -0.4386549, 0.0872523, -0.1360535, 0.1360439
8: -3.6494575, -1.5500991, -3.6494575, -1.5500991, -0.9167327, 0.9170370
9: -2.9068365, -1.1850188, -2.9068365, -1.1850188, -0.7912374, 0.7914890

Time for backsubstitution: 5.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 361
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 651
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 3525
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 810

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2656

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930727, upper bound: 0.0930756
time: 280.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930726, upper bound: 0.0930741
time: 252.07 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 538.68 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 538.68
Output dim: 5, lower bound: -0.0930521, upper bound: 0.0930721
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 538.68
Output dim: 5, lower bound: -0.0930521, upper bound: 0.0930721
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 538.68
Output dim: 5, lower bound: -0.0930942, upper bound: 0.0930864
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 538.68
Output dim: 5, lower bound: -0.0930943, upper bound: 0.0930861
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 538.68
Output dim: 5, lower bound: -0.0930531, upper bound: 0.0930804
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 538.68
Output dim: 5, lower bound: -0.0930723, upper bound: 0.0930584
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 538.68
Output dim: 5, lower bound: -0.0930727, upper bound: 0.0930756
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 538.68
Output dim: 5, lower bound: -0.0930726, upper bound: 0.0930741

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.2739444, -1.3582740, -3.2739444, -1.3582740, -1.5927122, 1.5926168
1: -3.1609457, -1.3756382, -3.1609457, -1.3756382, -1.2116175, 1.2115555
2: -1.0850873, -0.6183419, -1.0850873, -0.6183419, -0.2404671, 0.2404646
3: 0.3414853, 0.6887888, 0.3414853, 0.6887888, -0.2271308, 0.2271321
4: -4.2210059, -3.3748572, -4.2210059, -3.3748572, -0.2736189, 0.2736503
5: 1.8216184, 2.1860926, 1.8216184, 2.1860926, -0.1307591, 0.1307657
6: -4.4172969, -3.4960172, -4.4172969, -3.4960172, -0.1778645, 0.1778865
7: -0.4386549, 0.0872523, -0.4386549, 0.0872523, -0.1359915, 0.1359982
8: -3.6494575, -1.5500991, -3.6494575, -1.5500991, -0.9160578, 0.9157057
9: -2.9068365, -1.1850188, -2.9068365, -1.1850188, -0.7914208, 0.7912561

Time for backsubstitution: 5.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 361
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 651
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 3525
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 809

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3116

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930338, upper bound: 0.0930564
time: 32.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930378, upper bound: 0.0930525
time: 17.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.2739444, -1.3582740, -3.2739444, -1.3582740, -1.5926576, 1.5926192
1: -3.1609457, -1.3756382, -3.1609457, -1.3756382, -1.2115481, 1.2115600
2: -1.0850873, -0.6183419, -1.0850873, -0.6183419, -0.2404670, 0.2404646
3: 0.3414853, 0.6887888, 0.3414853, 0.6887888, -0.2271309, 0.2271321
4: -4.2210059, -3.3748572, -4.2210059, -3.3748572, -0.2736208, 0.2736090
5: 1.8216184, 2.1860926, 1.8216184, 2.1860926, -0.1307593, 0.1307632
6: -4.4172969, -3.4960172, -4.4172969, -3.4960172, -0.1778649, 0.1778746
7: -0.4386549, 0.0872523, -0.4386549, 0.0872523, -0.1359916, 0.1359957
8: -3.6494575, -1.5500991, -3.6494575, -1.5500991, -0.9158149, 0.9157161
9: -2.9068365, -1.1850188, -2.9068365, -1.1850188, -0.7912493, 0.7912657

Time for backsubstitution: 5.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 361
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 3525
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 651
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2116

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2209

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930252, upper bound: 0.0930533
time: 15.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930365, upper bound: 0.0930462
time: 6.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.2739444, -1.3582740, -3.2739444, -1.3582740, -1.5926099, 1.5926507
1: -3.1609457, -1.3756382, -3.1609457, -1.3756382, -1.2115630, 1.2115550
2: -1.0850873, -0.6183419, -1.0850873, -0.6183419, -0.2404615, 0.2404640
3: 0.3414853, 0.6887888, 0.3414853, 0.6887888, -0.2271334, 0.2271323
4: -4.2210059, -3.3748572, -4.2210059, -3.3748572, -0.2736092, 0.2736194
5: 1.8216184, 2.1860926, 1.8216184, 2.1860926, -0.1307568, 0.1307525
6: -4.4172969, -3.4960172, -4.4172969, -3.4960172, -0.1778659, 0.1778563
7: -0.4386549, 0.0872523, -0.4386549, 0.0872523, -0.1359958, 0.1359916
8: -3.6494575, -1.5500991, -3.6494575, -1.5500991, -0.9157138, 0.9158227
9: -2.9068365, -1.1850188, -2.9068365, -1.1850188, -0.7912662, 0.7912593

Time for backsubstitution: 5.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 3525
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 361
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 651
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2685

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 749

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930942, upper bound: 0.0930889
time: 71.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930942, upper bound: 0.0930865
time: 165.90 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.2739444, -1.3582740, -3.2739444, -1.3582740, -1.5926099, 1.5926507
1: -3.1609457, -1.3756382, -3.1609457, -1.3756382, -1.2115628, 1.2115555
2: -1.0850873, -0.6183419, -1.0850873, -0.6183419, -0.2404616, 0.2404638
3: 0.3414853, 0.6887888, 0.3414853, 0.6887888, -0.2271336, 0.2271321
4: -4.2210059, -3.3748572, -4.2210059, -3.3748572, -0.2736095, 0.2736192
5: 1.8216184, 2.1860926, 1.8216184, 2.1860926, -0.1307567, 0.1307527
6: -4.4172969, -3.4960172, -4.4172969, -3.4960172, -0.1778664, 0.1778557
7: -0.4386549, 0.0872523, -0.4386549, 0.0872523, -0.1359958, 0.1359916
8: -3.6494575, -1.5500991, -3.6494575, -1.5500991, -0.9157135, 0.9158230
9: -2.9068365, -1.1850188, -2.9068365, -1.1850188, -0.7912661, 0.7912594

Time for backsubstitution: 5.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 651
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 361
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3525
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 3024

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 777

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930920, upper bound: 0.0930844
time: 34.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930924, upper bound: 0.0930900
time: 26.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.2739444, -1.3582740, -3.2739444, -1.3582740, -1.5926203, 1.5925479
1: -3.1609457, -1.3756382, -3.1609457, -1.3756382, -1.2096033, 1.2093916
2: -1.0850873, -0.6183419, -1.0850873, -0.6183419, -0.2405018, 0.2405038
3: 0.3414853, 0.6887888, 0.3414853, 0.6887888, -0.2271579, 0.2271587
4: -4.2210059, -3.3748572, -4.2210059, -3.3748572, -0.2726468, 0.2727759
5: 1.8216184, 2.1860926, 1.8216184, 2.1860926, -0.1307659, 0.1307732
6: -4.4172969, -3.4960172, -4.4172969, -3.4960172, -0.1781121, 0.1781146
7: -0.4386549, 0.0872523, -0.4386549, 0.0872523, -0.1358587, 0.1358776
8: -3.6494575, -1.5500991, -3.6494575, -1.5500991, -0.9133069, 0.9127386
9: -2.9068365, -1.1850188, -2.9068365, -1.1850188, -0.7887332, 0.7882061

Time for backsubstitution: 5.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 651
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 3525
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 361
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2244

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2690

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930528, upper bound: 0.0930776
time: 25.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930528, upper bound: 0.0930777
time: 68.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.2739444, -1.3582740, -3.2739444, -1.3582740, -1.5925781, 1.5925903
1: -3.1609457, -1.3756382, -3.1609457, -1.3756382, -1.2094569, 1.2095380
2: -1.0850873, -0.6183419, -1.0850873, -0.6183419, -0.2405021, 0.2405035
3: 0.3414853, 0.6887888, 0.3414853, 0.6887888, -0.2271587, 0.2271579
4: -4.2210059, -3.3748572, -4.2210059, -3.3748572, -0.2727062, 0.2727166
5: 1.8216184, 2.1860926, 1.8216184, 2.1860926, -0.1307723, 0.1307668
6: -4.4172969, -3.4960172, -4.4172969, -3.4960172, -0.1781147, 0.1781120
7: -0.4386549, 0.0872523, -0.4386549, 0.0872523, -0.1358680, 0.1358683
8: -3.6494575, -1.5500991, -3.6494575, -1.5500991, -0.9130430, 0.9130025
9: -2.9068365, -1.1850188, -2.9068365, -1.1850188, -0.7884576, 0.7884816

Time for backsubstitution: 5.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 651
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 361
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 3525
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2602

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 105

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930670, upper bound: 0.0930573
time: 159.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930734, upper bound: 0.0930559
time: 16.08 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.2739444, -1.3582740, -3.2739444, -1.3582740, -1.5929382, 1.5929654
1: -3.1609457, -1.3756382, -3.1609457, -1.3756382, -1.2108555, 1.2109251
2: -1.0850873, -0.6183419, -1.0850873, -0.6183419, -0.2404266, 0.2404211
3: 0.3414853, 0.6887888, 0.3414853, 0.6887888, -0.2271601, 0.2271604
4: -4.2210059, -3.3748572, -4.2210059, -3.3748572, -0.2733170, 0.2732455
5: 1.8216184, 2.1860926, 1.8216184, 2.1860926, -0.1308040, 0.1308029
6: -4.4172969, -3.4960172, -4.4172969, -3.4960172, -0.1780985, 0.1780985
7: -0.4386549, 0.0872523, -0.4386549, 0.0872523, -0.1359426, 0.1359291
8: -3.6494575, -1.5500991, -3.6494575, -1.5500991, -0.9149626, 0.9152880
9: -2.9068365, -1.1850188, -2.9068365, -1.1850188, -0.7911520, 0.7914004

Time for backsubstitution: 5.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 3525
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 361
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 651
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 750

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 835

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930577, upper bound: 0.0930660
time: 51.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930644, upper bound: 0.0930625
time: 39.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.2739444, -1.3582740, -3.2739444, -1.3582740, -1.5929353, 1.5929682
1: -3.1609457, -1.3756382, -3.1609457, -1.3756382, -1.2108598, 1.2109210
2: -1.0850873, -0.6183419, -1.0850873, -0.6183419, -0.2404228, 0.2404249
3: 0.3414853, 0.6887888, 0.3414853, 0.6887888, -0.2271604, 0.2271601
4: -4.2210059, -3.3748572, -4.2210059, -3.3748572, -0.2733152, 0.2732472
5: 1.8216184, 2.1860926, 1.8216184, 2.1860926, -0.1308039, 0.1308030
6: -4.4172969, -3.4960172, -4.4172969, -3.4960172, -0.1780984, 0.1780986
7: -0.4386549, 0.0872523, -0.4386549, 0.0872523, -0.1359387, 0.1359330
8: -3.6494575, -1.5500991, -3.6494575, -1.5500991, -0.9149835, 0.9152670
9: -2.9068365, -1.1850188, -2.9068365, -1.1850188, -0.7911488, 0.7914037

Time for backsubstitution: 5.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 651
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 361
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 3525
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2513

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2084

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930720, upper bound: 0.0930780
time: 11.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930720, upper bound: 0.0930776
time: 10.20 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 28.04 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.04
Output dim: 5, lower bound: -0.0930338, upper bound: 0.0930564
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.04
Output dim: 5, lower bound: -0.0930378, upper bound: 0.0930525
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.04
Output dim: 5, lower bound: -0.0930252, upper bound: 0.0930533
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.04
Output dim: 5, lower bound: -0.0930365, upper bound: 0.0930462
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.04
Output dim: 5, lower bound: -0.0930942, upper bound: 0.0930889
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.04
Output dim: 5, lower bound: -0.0930942, upper bound: 0.0930865
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.04
Output dim: 5, lower bound: -0.0930920, upper bound: 0.0930844
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.04
Output dim: 5, lower bound: -0.0930924, upper bound: 0.0930900
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.04
Output dim: 5, lower bound: -0.0930528, upper bound: 0.0930776
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.04
Output dim: 5, lower bound: -0.0930528, upper bound: 0.0930777
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.04
Output dim: 5, lower bound: -0.0930670, upper bound: 0.0930573
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.04
Output dim: 5, lower bound: -0.0930734, upper bound: 0.0930559
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.04
Output dim: 5, lower bound: -0.0930577, upper bound: 0.0930660
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.04
Output dim: 5, lower bound: -0.0930644, upper bound: 0.0930625
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.04
Output dim: 5, lower bound: -0.0930720, upper bound: 0.0930780
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.04
Output dim: 5, lower bound: -0.0930720, upper bound: 0.0930776

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.2739444, -1.3582740, -3.2739444, -1.3582740, -1.5916076, 1.5913646
1: -3.1609457, -1.3756382, -3.1609457, -1.3756382, -1.2058206, 1.2053232
2: -1.0850873, -0.6183419, -1.0850873, -0.6183419, -0.2404359, 0.2404291
3: 0.3414853, 0.6887888, 0.3414853, 0.6887888, -0.2271281, 0.2271294
4: -4.2210059, -3.3748572, -4.2210059, -3.3748572, -0.2680966, 0.2683719
5: 1.8216184, 2.1860926, 1.8216184, 2.1860926, -0.1305215, 0.1305395
6: -4.4172969, -3.4960172, -4.4172969, -3.4960172, -0.1772960, 0.1773359
7: -0.4386549, 0.0872523, -0.4386549, 0.0872523, -0.1358351, 0.1358518
8: -3.6494575, -1.5500991, -3.6494575, -1.5500991, -0.9032944, 0.9021434
9: -2.9068365, -1.1850188, -2.9068365, -1.1850188, -0.7784073, 0.7774310

Time for backsubstitution: 5.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 361
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 651
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 3525
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2393

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2131

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930272, upper bound: 0.0930549
time: 13.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930338, upper bound: 0.0930458
time: 26.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.2739444, -1.3582740, -3.2739444, -1.3582740, -1.5914602, 1.5915122
1: -3.1609457, -1.3756382, -3.1609457, -1.3756382, -1.2053854, 1.2057581
2: -1.0850873, -0.6183419, -1.0850873, -0.6183419, -0.2404317, 0.2404333
3: 0.3414853, 0.6887888, 0.3414853, 0.6887888, -0.2271281, 0.2271293
4: -4.2210059, -3.3748572, -4.2210059, -3.3748572, -0.2683405, 0.2681279
5: 1.8216184, 2.1860926, 1.8216184, 2.1860926, -0.1305329, 0.1305281
6: -4.4172969, -3.4960172, -4.4172969, -3.4960172, -0.1773139, 0.1773180
7: -0.4386549, 0.0872523, -0.4386549, 0.0872523, -0.1358450, 0.1358418
8: -3.6494575, -1.5500991, -3.6494575, -1.5500991, -0.9024955, 0.9029417
9: -2.9068365, -1.1850188, -2.9068365, -1.1850188, -0.7775958, 0.7782420

Time for backsubstitution: 5.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 651
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 3525
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 361
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2582

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 155

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930153, upper bound: 0.0930240
time: 23.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930100, upper bound: 0.0930324
time: 17.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.2739444, -1.3582740, -3.2739444, -1.3582740, -1.5925778, 1.5925102
1: -3.1609457, -1.3756382, -3.1609457, -1.3756382, -1.2113082, 1.2111547
2: -1.0850873, -0.6183419, -1.0850873, -0.6183419, -0.2404659, 0.2404649
3: 0.3414853, 0.6887888, 0.3414853, 0.6887888, -0.2271306, 0.2271314
4: -4.2210059, -3.3748572, -4.2210059, -3.3748572, -0.2734429, 0.2735066
5: 1.8216184, 2.1860926, 1.8216184, 2.1860926, -0.1307424, 0.1307486
6: -4.4172969, -3.4960172, -4.4172969, -3.4960172, -0.1778634, 0.1778708
7: -0.4386549, 0.0872523, -0.4386549, 0.0872523, -0.1359651, 0.1359832
8: -3.6494575, -1.5500991, -3.6494575, -1.5500991, -0.9154522, 0.9149826
9: -2.9068365, -1.1850188, -2.9068365, -1.1850188, -0.7910548, 0.7906742

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 651
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 3525
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 361
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2992

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2116

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930149, upper bound: 0.0930431
time: 163.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930263, upper bound: 0.0930468
time: 6.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.2739444, -1.3582740, -3.2739444, -1.3582740, -1.5925480, 1.5925400
1: -3.1609457, -1.3756382, -3.1609457, -1.3756382, -1.2111425, 1.2113206
2: -1.0850873, -0.6183419, -1.0850873, -0.6183419, -0.2404672, 0.2404635
3: 0.3414853, 0.6887888, 0.3414853, 0.6887888, -0.2271301, 0.2271318
4: -4.2210059, -3.3748572, -4.2210059, -3.3748572, -0.2735183, 0.2734311
5: 1.8216184, 2.1860926, 1.8216184, 2.1860926, -0.1307448, 0.1307462
6: -4.4172969, -3.4960172, -4.4172969, -3.4960172, -0.1778611, 0.1778731
7: -0.4386549, 0.0872523, -0.4386549, 0.0872523, -0.1359791, 0.1359691
8: -3.6494575, -1.5500991, -3.6494575, -1.5500991, -0.9150815, 0.9153533
9: -2.9068365, -1.1850188, -2.9068365, -1.1850188, -0.7906579, 0.7910712

Time for backsubstitution: 6.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 651
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 3525
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 361
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2237

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2245

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930368, upper bound: 0.0930390
time: 371.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930368, upper bound: 0.0930417
time: 288.90 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 666.26 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 666.26
Output dim: 5, lower bound: -0.0930272, upper bound: 0.0930549
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 666.26
Output dim: 5, lower bound: -0.0930338, upper bound: 0.0930458
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 666.26
Output dim: 5, lower bound: -0.0930153, upper bound: 0.0930240
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 666.26
Output dim: 5, lower bound: -0.0930100, upper bound: 0.0930324
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 666.26
Output dim: 5, lower bound: -0.0930149, upper bound: 0.0930431
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 666.26
Output dim: 5, lower bound: -0.0930263, upper bound: 0.0930468
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 666.26
Output dim: 5, lower bound: -0.0930368, upper bound: 0.0930390
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 666.26
Output dim: 5, lower bound: -0.0930368, upper bound: 0.0930417
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 666.26
Output dim: 5, lower bound: -0.0930942, upper bound: 0.0930889
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 666.26
Output dim: 5, lower bound: -0.0930942, upper bound: 0.0930865
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 666.26
Output dim: 5, lower bound: -0.0930920, upper bound: 0.0930844
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 666.26
Output dim: 5, lower bound: -0.0930924, upper bound: 0.0930900
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 666.26
Output dim: 5, lower bound: -0.0930528, upper bound: 0.0930776
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 666.26
Output dim: 5, lower bound: -0.0930528, upper bound: 0.0930777
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 666.26
Output dim: 5, lower bound: -0.0930670, upper bound: 0.0930573
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 666.26
Output dim: 5, lower bound: -0.0930734, upper bound: 0.0930559
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 666.26
Output dim: 5, lower bound: -0.0930577, upper bound: 0.0930660
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 666.26
Output dim: 5, lower bound: -0.0930644, upper bound: 0.0930625
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 666.26
Output dim: 5, lower bound: -0.0930720, upper bound: 0.0930780
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 666.26
Output dim: 5, lower bound: -0.0930720, upper bound: 0.0930776

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 84.45 + 3586.73 = 3671.18 seconds
