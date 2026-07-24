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
execution time: IAR + RelationalAnalysis = 7.88 + 77.26 = 85.14 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0930953, upper bound: 0.0931023

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 361
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 651
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3525

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2171

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930541, upper bound: 0.0930935
time: 6.63 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930830, upper bound: 0.0930596
time: 35.13 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 41.84 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 41.84
Output dim: 5, lower bound: -0.0930541, upper bound: 0.0930935
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 41.84
Output dim: 5, lower bound: -0.0930830, upper bound: 0.0930596

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -3.2739444, -1.3582740, -3.2739444, -1.3582740, -1.5921233, 1.5920775
1: -3.1609457, -1.3756382, -3.1609457, -1.3756382, -1.2106853, 1.2106855
2: -1.0850873, -0.6183419, -1.0850873, -0.6183419, -0.2402166, 0.2402184
3: 0.3414853, 0.6887888, 0.3414853, 0.6887888, -0.2270328, 0.2270328
4: -4.2210059, -3.3748572, -4.2210059, -3.3748572, -0.2737076, 0.2737100
5: 1.8216184, 2.1860926, 1.8216184, 2.1860926, -0.1305619, 0.1305618
6: -4.4172969, -3.4960172, -4.4172969, -3.4960172, -0.1775436, 0.1775516
7: -0.4386549, 0.0872523, -0.4386549, 0.0872523, -0.1361867, 0.1361868
8: -3.6494575, -1.5500991, -3.6494575, -1.5500991, -0.9153240, 0.9151837
9: -2.9068365, -1.1850188, -2.9068365, -1.1850188, -0.7903233, 0.7903198

Time for backsubstitution: 6.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 361
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 651
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3525

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2156

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0929891, upper bound: 0.0930857
time: 41.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930514, upper bound: 0.0930266
time: 8.93 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -3.2739444, -1.3582740, -3.2739444, -1.3582740, -1.5920775, 1.5921233
1: -3.1609457, -1.3756382, -3.1609457, -1.3756382, -1.2106853, 1.2106853
2: -1.0850873, -0.6183419, -1.0850873, -0.6183419, -0.2402184, 0.2402166
3: 0.3414853, 0.6887888, 0.3414853, 0.6887888, -0.2270328, 0.2270328
4: -4.2210059, -3.3748572, -4.2210059, -3.3748572, -0.2737101, 0.2737076
5: 1.8216184, 2.1860926, 1.8216184, 2.1860926, -0.1305617, 0.1305619
6: -4.4172969, -3.4960172, -4.4172969, -3.4960172, -0.1775516, 0.1775436
7: -0.4386549, 0.0872523, -0.4386549, 0.0872523, -0.1361868, 0.1361867
8: -3.6494575, -1.5500991, -3.6494575, -1.5500991, -0.9151837, 0.9153241
9: -2.9068365, -1.1850188, -2.9068365, -1.1850188, -0.7903199, 0.7903233

Time for backsubstitution: 6.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 361
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 651
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3525

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2156

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930165, upper bound: 0.0929914
time: 164.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930833, upper bound: 0.0929972
time: 12.92 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 183.73 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 183.73
Output dim: 5, lower bound: -0.0929891, upper bound: 0.0930857
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 183.73
Output dim: 5, lower bound: -0.0930514, upper bound: 0.0930266
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 183.73
Output dim: 5, lower bound: -0.0930165, upper bound: 0.0929914
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 183.73
Output dim: 5, lower bound: -0.0930833, upper bound: 0.0929972

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.2739444, -1.3582740, -3.2739444, -1.3582740, -1.5914996, 1.5913842
1: -3.1609457, -1.3756382, -3.1609457, -1.3756382, -1.2099782, 1.2099731
2: -1.0850873, -0.6183419, -1.0850873, -0.6183419, -0.2400821, 0.2400696
3: 0.3414853, 0.6887888, 0.3414853, 0.6887888, -0.2269481, 0.2269564
4: -4.2210059, -3.3748572, -4.2210059, -3.3748572, -0.2736874, 0.2736899
5: 1.8216184, 2.1860926, 1.8216184, 2.1860926, -0.1304127, 0.1304184
6: -4.4172969, -3.4960172, -4.4172969, -3.4960172, -0.1772437, 0.1772779
7: -0.4386549, 0.0872523, -0.4386549, 0.0872523, -0.1361665, 0.1361655
8: -3.6494575, -1.5500991, -3.6494575, -1.5500991, -0.9141223, 0.9138455
9: -2.9068365, -1.1850188, -2.9068365, -1.1850188, -0.7892877, 0.7892444

Time for backsubstitution: 5.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 361
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 651
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3525

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2170

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0929493, upper bound: 0.0930579
time: 52.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0929507, upper bound: 0.0930467
time: 7.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.2739444, -1.3582740, -3.2739444, -1.3582740, -1.5914299, 1.5914550
1: -3.1609457, -1.3756382, -3.1609457, -1.3756382, -1.2099727, 1.2099786
2: -1.0850873, -0.6183419, -1.0850873, -0.6183419, -0.2400678, 0.2400839
3: 0.3414853, 0.6887888, 0.3414853, 0.6887888, -0.2269565, 0.2269481
4: -4.2210059, -3.3748572, -4.2210059, -3.3748572, -0.2736875, 0.2736899
5: 1.8216184, 2.1860926, 1.8216184, 2.1860926, -0.1304189, 0.1304125
6: -4.4172969, -3.4960172, -4.4172969, -3.4960172, -0.1772699, 0.1772518
7: -0.4386549, 0.0872523, -0.4386549, 0.0872523, -0.1361654, 0.1361666
8: -3.6494575, -1.5500991, -3.6494575, -1.5500991, -0.9139858, 0.9139843
9: -2.9068365, -1.1850188, -2.9068365, -1.1850188, -0.7892479, 0.7892857

Time for backsubstitution: 5.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 361
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 651
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3525

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2170

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930146, upper bound: 0.0929969
time: 6.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930149, upper bound: 0.0929806
time: 152.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.2739444, -1.3582740, -3.2739444, -1.3582740, -1.5914550, 1.5914299
1: -3.1609457, -1.3756382, -3.1609457, -1.3756382, -1.2099787, 1.2099729
2: -1.0850873, -0.6183419, -1.0850873, -0.6183419, -0.2400839, 0.2400678
3: 0.3414853, 0.6887888, 0.3414853, 0.6887888, -0.2269481, 0.2269565
4: -4.2210059, -3.3748572, -4.2210059, -3.3748572, -0.2736899, 0.2736875
5: 1.8216184, 2.1860926, 1.8216184, 2.1860926, -0.1304125, 0.1304189
6: -4.4172969, -3.4960172, -4.4172969, -3.4960172, -0.1772518, 0.1772699
7: -0.4386549, 0.0872523, -0.4386549, 0.0872523, -0.1361666, 0.1361654
8: -3.6494575, -1.5500991, -3.6494575, -1.5500991, -0.9139843, 0.9139858
9: -2.9068365, -1.1850188, -2.9068365, -1.1850188, -0.7892857, 0.7892479

Time for backsubstitution: 5.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 361
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 651
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3525

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 2170

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0929750, upper bound: 0.0930211
time: 13.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0929872, upper bound: 0.0930223
time: 32.91 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.2739444, -1.3582740, -3.2739444, -1.3582740, -1.5913842, 1.5914996
1: -3.1609457, -1.3756382, -3.1609457, -1.3756382, -1.2099730, 1.2099781
2: -1.0850873, -0.6183419, -1.0850873, -0.6183419, -0.2400696, 0.2400821
3: 0.3414853, 0.6887888, 0.3414853, 0.6887888, -0.2269564, 0.2269481
4: -4.2210059, -3.3748572, -4.2210059, -3.3748572, -0.2736899, 0.2736875
5: 1.8216184, 2.1860926, 1.8216184, 2.1860926, -0.1304184, 0.1304127
6: -4.4172969, -3.4960172, -4.4172969, -3.4960172, -0.1772779, 0.1772437
7: -0.4386549, 0.0872523, -0.4386549, 0.0872523, -0.1361655, 0.1361665
8: -3.6494575, -1.5500991, -3.6494575, -1.5500991, -0.9138455, 0.9141223
9: -2.9068365, -1.1850188, -2.9068365, -1.1850188, -0.7892443, 0.7892877

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 361
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 651
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3525

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2170

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930391, upper bound: 0.0929576
time: 11.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930504, upper bound: 0.0929530
time: 210.50 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 227.57 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 227.57
Output dim: 5, lower bound: -0.0929493, upper bound: 0.0930579
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 227.57
Output dim: 5, lower bound: -0.0929507, upper bound: 0.0930467
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 227.57
Output dim: 5, lower bound: -0.0930146, upper bound: 0.0929969
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 227.57
Output dim: 5, lower bound: -0.0930149, upper bound: 0.0929806
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 227.57
Output dim: 5, lower bound: -0.0929750, upper bound: 0.0930211
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 227.57
Output dim: 5, lower bound: -0.0929872, upper bound: 0.0930223
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 227.57
Output dim: 5, lower bound: -0.0930391, upper bound: 0.0929576
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 227.57
Output dim: 5, lower bound: -0.0930504, upper bound: 0.0929530

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.2739444, -1.3582740, -3.2739444, -1.3582740, -1.5913420, 1.5911579
1: -3.1609457, -1.3756382, -3.1609457, -1.3756382, -1.2096388, 1.2096336
2: -1.0850873, -0.6183419, -1.0850873, -0.6183419, -0.2400380, 0.2400261
3: 0.3414853, 0.6887888, 0.3414853, 0.6887888, -0.2269163, 0.2269266
4: -4.2210059, -3.3748572, -4.2210059, -3.3748572, -0.2736672, 0.2736710
5: 1.8216184, 2.1860926, 1.8216184, 2.1860926, -0.1303543, 0.1303599
6: -4.4172969, -3.4960172, -4.4172969, -3.4960172, -0.1771545, 0.1772012
7: -0.4386549, 0.0872523, -0.4386549, 0.0872523, -0.1361631, 0.1361624
8: -3.6494575, -1.5500991, -3.6494575, -1.5500991, -0.9136804, 0.9132075
9: -2.9068365, -1.1850188, -2.9068365, -1.1850188, -0.7887584, 0.7887137

Time for backsubstitution: 6.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 361
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 651
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3525

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0929231, upper bound: 0.0930534
time: 89.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0929492, upper bound: 0.0930319
time: 11.94 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.2739444, -1.3582740, -3.2739444, -1.3582740, -1.5912735, 1.5912030
1: -3.1609457, -1.3756382, -3.1609457, -1.3756382, -1.2096388, 1.2096319
2: -1.0850873, -0.6183419, -1.0850873, -0.6183419, -0.2400386, 0.2400247
3: 0.3414853, 0.6887888, 0.3414853, 0.6887888, -0.2269162, 0.2269246
4: -4.2210059, -3.3748572, -4.2210059, -3.3748572, -0.2736680, 0.2736696
5: 1.8216184, 2.1860926, 1.8216184, 2.1860926, -0.1303540, 0.1303600
6: -4.4172969, -3.4960172, -4.4172969, -3.4960172, -0.1771615, 0.1771887
7: -0.4386549, 0.0872523, -0.4386549, 0.0872523, -0.1361633, 0.1361621
8: -3.6494575, -1.5500991, -3.6494575, -1.5500991, -0.9134843, 0.9133995
9: -2.9068365, -1.1850188, -2.9068365, -1.1850188, -0.7887570, 0.7887125

Time for backsubstitution: 5.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 361
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 651
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3525

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0929244, upper bound: 0.0930166
time: 193.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0929499, upper bound: 0.0930191
time: 9.87 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.2739444, -1.3582740, -3.2739444, -1.3582740, -1.5912724, 1.5912287
1: -3.1609457, -1.3756382, -3.1609457, -1.3756382, -1.2096336, 1.2096393
2: -1.0850873, -0.6183419, -1.0850873, -0.6183419, -0.2400237, 0.2400404
3: 0.3414853, 0.6887888, 0.3414853, 0.6887888, -0.2269247, 0.2269184
4: -4.2210059, -3.3748572, -4.2210059, -3.3748572, -0.2736672, 0.2736710
5: 1.8216184, 2.1860926, 1.8216184, 2.1860926, -0.1303605, 0.1303541
6: -4.4172969, -3.4960172, -4.4172969, -3.4960172, -0.1771807, 0.1771752
7: -0.4386549, 0.0872523, -0.4386549, 0.0872523, -0.1361621, 0.1361635
8: -3.6494575, -1.5500991, -3.6494575, -1.5500991, -0.9135439, 0.9133463
9: -2.9068365, -1.1850188, -2.9068365, -1.1850188, -0.7887193, 0.7887551

Time for backsubstitution: 6.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 361
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 651
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3525

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0929845, upper bound: 0.0929958
time: 19.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930122, upper bound: 0.0929670
time: 95.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.2739444, -1.3582740, -3.2739444, -1.3582740, -1.5912039, 1.5912735
1: -3.1609457, -1.3756382, -3.1609457, -1.3756382, -1.2096334, 1.2096374
2: -1.0850873, -0.6183419, -1.0850873, -0.6183419, -0.2400243, 0.2400390
3: 0.3414853, 0.6887888, 0.3414853, 0.6887888, -0.2269246, 0.2269163
4: -4.2210059, -3.3748572, -4.2210059, -3.3748572, -0.2736680, 0.2736696
5: 1.8216184, 2.1860926, 1.8216184, 2.1860926, -0.1303600, 0.1303541
6: -4.4172969, -3.4960172, -4.4172969, -3.4960172, -0.1771877, 0.1771625
7: -0.4386549, 0.0872523, -0.4386549, 0.0872523, -0.1361623, 0.1361632
8: -3.6494575, -1.5500991, -3.6494575, -1.5500991, -0.9133480, 0.9135379
9: -2.9068365, -1.1850188, -2.9068365, -1.1850188, -0.7887172, 0.7887533

Time for backsubstitution: 5.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 361
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 651
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3525

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0929884, upper bound: 0.0929797
time: 15.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930130, upper bound: 0.0929534
time: 36.43 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.2739444, -1.3582740, -3.2739444, -1.3582740, -1.5912738, 1.5912037
1: -3.1609457, -1.3756382, -3.1609457, -1.3756382, -1.2096374, 1.2096334
2: -1.0850873, -0.6183419, -1.0850873, -0.6183419, -0.2400390, 0.2400243
3: 0.3414853, 0.6887888, 0.3414853, 0.6887888, -0.2269163, 0.2269246
4: -4.2210059, -3.3748572, -4.2210059, -3.3748572, -0.2736695, 0.2736680
5: 1.8216184, 2.1860926, 1.8216184, 2.1860926, -0.1303541, 0.1303601
6: -4.4172969, -3.4960172, -4.4172969, -3.4960172, -0.1771625, 0.1771877
7: -0.4386549, 0.0872523, -0.4386549, 0.0872523, -0.1361632, 0.1361623
8: -3.6494575, -1.5500991, -3.6494575, -1.5500991, -0.9135380, 0.9133478
9: -2.9068365, -1.1850188, -2.9068365, -1.1850188, -0.7887532, 0.7887173

Time for backsubstitution: 6.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 361
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 651
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3525

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0929508, upper bound: 0.0930188
time: 14.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0929731, upper bound: 0.0929908
time: 62.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.2739444, -1.3582740, -3.2739444, -1.3582740, -1.5912290, 1.5912724
1: -3.1609457, -1.3756382, -3.1609457, -1.3756382, -1.2096393, 1.2096336
2: -1.0850873, -0.6183419, -1.0850873, -0.6183419, -0.2400404, 0.2400237
3: 0.3414853, 0.6887888, 0.3414853, 0.6887888, -0.2269184, 0.2269247
4: -4.2210059, -3.3748572, -4.2210059, -3.3748572, -0.2736710, 0.2736672
5: 1.8216184, 2.1860926, 1.8216184, 2.1860926, -0.1303540, 0.1303605
6: -4.4172969, -3.4960172, -4.4172969, -3.4960172, -0.1771752, 0.1771807
7: -0.4386549, 0.0872523, -0.4386549, 0.0872523, -0.1361635, 0.1361621
8: -3.6494575, -1.5500991, -3.6494575, -1.5500991, -0.9133463, 0.9135439
9: -2.9068365, -1.1850188, -2.9068365, -1.1850188, -0.7887551, 0.7887192

Time for backsubstitution: 6.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 361
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 651
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3525

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0929638, upper bound: 0.0930214
time: 44.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0929881, upper bound: 0.0929921
time: 10.05 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.2739444, -1.3582740, -3.2739444, -1.3582740, -1.5912032, 1.5912733
1: -3.1609457, -1.3756382, -3.1609457, -1.3756382, -1.2096319, 1.2096388
2: -1.0850873, -0.6183419, -1.0850873, -0.6183419, -0.2400247, 0.2400386
3: 0.3414853, 0.6887888, 0.3414853, 0.6887888, -0.2269246, 0.2269162
4: -4.2210059, -3.3748572, -4.2210059, -3.3748572, -0.2736695, 0.2736680
5: 1.8216184, 2.1860926, 1.8216184, 2.1860926, -0.1303600, 0.1303540
6: -4.4172969, -3.4960172, -4.4172969, -3.4960172, -0.1771887, 0.1771615
7: -0.4386549, 0.0872523, -0.4386549, 0.0872523, -0.1361622, 0.1361633
8: -3.6494575, -1.5500991, -3.6494575, -1.5500991, -0.9133995, 0.9134843
9: -2.9068365, -1.1850188, -2.9068365, -1.1850188, -0.7887124, 0.7887570

Time for backsubstitution: 6.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 361
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 651
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3525

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930100, upper bound: 0.0929263
time: 114.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930366, upper bound: 0.0929319
time: 10.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.2739444, -1.3582740, -3.2739444, -1.3582740, -1.5911582, 1.5913420
1: -3.1609457, -1.3756382, -3.1609457, -1.3756382, -1.2096336, 1.2096388
2: -1.0850873, -0.6183419, -1.0850873, -0.6183419, -0.2400261, 0.2400380
3: 0.3414853, 0.6887888, 0.3414853, 0.6887888, -0.2269267, 0.2269163
4: -4.2210059, -3.3748572, -4.2210059, -3.3748572, -0.2736710, 0.2736672
5: 1.8216184, 2.1860926, 1.8216184, 2.1860926, -0.1303599, 0.1303543
6: -4.4172969, -3.4960172, -4.4172969, -3.4960172, -0.1772012, 0.1771545
7: -0.4386549, 0.0872523, -0.4386549, 0.0872523, -0.1361624, 0.1361631
8: -3.6494575, -1.5500991, -3.6494575, -1.5500991, -0.9132075, 0.9136804
9: -2.9068365, -1.1850188, -2.9068365, -1.1850188, -0.7887137, 0.7887585

Time for backsubstitution: 6.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 361
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 651
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3525

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930247, upper bound: 0.0929540
time: 95.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930510, upper bound: 0.0929273
time: 16.67 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 119.02 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 119.02
Output dim: 5, lower bound: -0.0929231, upper bound: 0.0930534
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 119.02
Output dim: 5, lower bound: -0.0929492, upper bound: 0.0930319
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 119.02
Output dim: 5, lower bound: -0.0929244, upper bound: 0.0930166
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 119.02
Output dim: 5, lower bound: -0.0929499, upper bound: 0.0930191
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 119.02
Output dim: 5, lower bound: -0.0929845, upper bound: 0.0929958
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 119.02
Output dim: 5, lower bound: -0.0930122, upper bound: 0.0929670
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 119.02
Output dim: 5, lower bound: -0.0929884, upper bound: 0.0929797
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 119.02
Output dim: 5, lower bound: -0.0930130, upper bound: 0.0929534
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 119.02
Output dim: 5, lower bound: -0.0929508, upper bound: 0.0930188
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 119.02
Output dim: 5, lower bound: -0.0929731, upper bound: 0.0929908
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 119.02
Output dim: 5, lower bound: -0.0929638, upper bound: 0.0930214
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 119.02
Output dim: 5, lower bound: -0.0929881, upper bound: 0.0929921
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 119.02
Output dim: 5, lower bound: -0.0930100, upper bound: 0.0929263
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 119.02
Output dim: 5, lower bound: -0.0930366, upper bound: 0.0929319
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 119.02
Output dim: 5, lower bound: -0.0930247, upper bound: 0.0929540
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 119.02
Output dim: 5, lower bound: -0.0930510, upper bound: 0.0929273

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.2739444, -1.3582740, -3.2739444, -1.3582740, -1.5908558, 1.5906458
1: -3.1609457, -1.3756382, -3.1609457, -1.3756382, -1.2089527, 1.2089503
2: -1.0850873, -0.6183419, -1.0850873, -0.6183419, -0.2399634, 0.2399475
3: 0.3414853, 0.6887888, 0.3414853, 0.6887888, -0.2269040, 0.2269157
4: -4.2210059, -3.3748572, -4.2210059, -3.3748572, -0.2736154, 0.2736217
5: 1.8216184, 2.1860926, 1.8216184, 2.1860926, -0.1302737, 0.1302794
6: -4.4172969, -3.4960172, -4.4172969, -3.4960172, -0.1770513, 0.1770987
7: -0.4386549, 0.0872523, -0.4386549, 0.0872523, -0.1361903, 0.1361897
8: -3.6494575, -1.5500991, -3.6494575, -1.5500991, -0.9124442, 0.9119059
9: -2.9068365, -1.1850188, -2.9068365, -1.1850188, -0.7875900, 0.7875311

Time for backsubstitution: 6.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 361
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 651
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3525

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2605

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0928885, upper bound: 0.0930079
time: 7.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0928668, upper bound: 0.0930169
time: 9.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.2739444, -1.3582740, -3.2739444, -1.3582740, -1.5908298, 1.5906725
1: -3.1609457, -1.3756382, -3.1609457, -1.3756382, -1.2089558, 1.2089477
2: -1.0850873, -0.6183419, -1.0850873, -0.6183419, -0.2399594, 0.2399516
3: 0.3414853, 0.6887888, 0.3414853, 0.6887888, -0.2269053, 0.2269144
4: -4.2210059, -3.3748572, -4.2210059, -3.3748572, -0.2736180, 0.2736191
5: 1.8216184, 2.1860926, 1.8216184, 2.1860926, -0.1302735, 0.1302793
6: -4.4172969, -3.4960172, -4.4172969, -3.4960172, -0.1770520, 0.1770981
7: -0.4386549, 0.0872523, -0.4386549, 0.0872523, -0.1361904, 0.1361896
8: -3.6494575, -1.5500991, -3.6494575, -1.5500991, -0.9123787, 0.9119734
9: -2.9068365, -1.1850188, -2.9068365, -1.1850188, -0.7875760, 0.7875464

Time for backsubstitution: 6.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 361
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 651
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3525

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2605

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0929110, upper bound: 0.0929792
time: 8.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0928932, upper bound: 0.0929916
time: 87.08 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.2739444, -1.3582740, -3.2739444, -1.3582740, -1.5907874, 1.5906909
1: -3.1609457, -1.3756382, -3.1609457, -1.3756382, -1.2089527, 1.2089489
2: -1.0850873, -0.6183419, -1.0850873, -0.6183419, -0.2399640, 0.2399462
3: 0.3414853, 0.6887888, 0.3414853, 0.6887888, -0.2269039, 0.2269136
4: -4.2210059, -3.3748572, -4.2210059, -3.3748572, -0.2736161, 0.2736204
5: 1.8216184, 2.1860926, 1.8216184, 2.1860926, -0.1302734, 0.1302795
6: -4.4172969, -3.4960172, -4.4172969, -3.4960172, -0.1770583, 0.1770861
7: -0.4386549, 0.0872523, -0.4386549, 0.0872523, -0.1361905, 0.1361894
8: -3.6494575, -1.5500991, -3.6494575, -1.5500991, -0.9122483, 0.9120978
9: -2.9068365, -1.1850188, -2.9068365, -1.1850188, -0.7875891, 0.7875299

Time for backsubstitution: 6.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 361
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 651
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3525

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 2605

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0928882, upper bound: 0.0929844
time: 16.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0928681, upper bound: 0.0929143
time: 43.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.2739444, -1.3582740, -3.2739444, -1.3582740, -1.5907614, 1.5907171
1: -3.1609457, -1.3756382, -3.1609457, -1.3756382, -1.2089555, 1.2089458
2: -1.0850873, -0.6183419, -1.0850873, -0.6183419, -0.2399600, 0.2399502
3: 0.3414853, 0.6887888, 0.3414853, 0.6887888, -0.2269052, 0.2269123
4: -4.2210059, -3.3748572, -4.2210059, -3.3748572, -0.2736188, 0.2736177
5: 1.8216184, 2.1860926, 1.8216184, 2.1860926, -0.1302732, 0.1302795
6: -4.4172969, -3.4960172, -4.4172969, -3.4960172, -0.1770590, 0.1770855
7: -0.4386549, 0.0872523, -0.4386549, 0.0872523, -0.1361906, 0.1361893
8: -3.6494575, -1.5500991, -3.6494575, -1.5500991, -0.9121826, 0.9121646
9: -2.9068365, -1.1850188, -2.9068365, -1.1850188, -0.7875745, 0.7875439

Time for backsubstitution: 6.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 361
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 651
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3525

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2605

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0929111, upper bound: 0.0929649
time: 7.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0928931, upper bound: 0.0929834
time: 8.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.2739444, -1.3582740, -3.2739444, -1.3582740, -1.5907862, 1.5907166
1: -3.1609457, -1.3756382, -3.1609457, -1.3756382, -1.2089474, 1.2089560
2: -1.0850873, -0.6183419, -1.0850873, -0.6183419, -0.2399492, 0.2399618
3: 0.3414853, 0.6887888, 0.3414853, 0.6887888, -0.2269124, 0.2269073
4: -4.2210059, -3.3748572, -4.2210059, -3.3748572, -0.2736154, 0.2736217
5: 1.8216184, 2.1860926, 1.8216184, 2.1860926, -0.1302799, 0.1302732
6: -4.4172969, -3.4960172, -4.4172969, -3.4960172, -0.1770775, 0.1770726
7: -0.4386549, 0.0872523, -0.4386549, 0.0872523, -0.1361893, 0.1361908
8: -3.6494575, -1.5500991, -3.6494575, -1.5500991, -0.9123079, 0.9120446
9: -2.9068365, -1.1850188, -2.9068365, -1.1850188, -0.7875497, 0.7875725

Time for backsubstitution: 6.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 361
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 651
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3525

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2605

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0929447, upper bound: 0.0929353
time: 145.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0929318, upper bound: 0.0929580
time: 12.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.2739444, -1.3582740, -3.2739444, -1.3582740, -1.5907602, 1.5907433
1: -3.1609457, -1.3756382, -3.1609457, -1.3756382, -1.2089503, 1.2089536
2: -1.0850873, -0.6183419, -1.0850873, -0.6183419, -0.2399452, 0.2399659
3: 0.3414853, 0.6887888, 0.3414853, 0.6887888, -0.2269137, 0.2269062
4: -4.2210059, -3.3748572, -4.2210059, -3.3748572, -0.2736180, 0.2736191
5: 1.8216184, 2.1860926, 1.8216184, 2.1860926, -0.1302801, 0.1302735
6: -4.4172969, -3.4960172, -4.4172969, -3.4960172, -0.1770782, 0.1770720
7: -0.4386549, 0.0872523, -0.4386549, 0.0872523, -0.1361894, 0.1361906
8: -3.6494575, -1.5500991, -3.6494575, -1.5500991, -0.9122422, 0.9121121
9: -2.9068365, -1.1850188, -2.9068365, -1.1850188, -0.7875367, 0.7875888

Time for backsubstitution: 6.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 361
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 651
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3525

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2605

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0929716, upper bound: 0.0929160
time: 109.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0929586, upper bound: 0.0929358
time: 12.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.2739444, -1.3582740, -3.2739444, -1.3582740, -1.5907178, 1.5907614
1: -3.1609457, -1.3756382, -3.1609457, -1.3756382, -1.2089472, 1.2089541
2: -1.0850873, -0.6183419, -1.0850873, -0.6183419, -0.2399497, 0.2399605
3: 0.3414853, 0.6887888, 0.3414853, 0.6887888, -0.2269123, 0.2269052
4: -4.2210059, -3.3748572, -4.2210059, -3.3748572, -0.2736161, 0.2736203
5: 1.8216184, 2.1860926, 1.8216184, 2.1860926, -0.1302795, 0.1302733
6: -4.4172969, -3.4960172, -4.4172969, -3.4960172, -0.1770845, 0.1770600
7: -0.4386549, 0.0872523, -0.4386549, 0.0872523, -0.1361894, 0.1361905
8: -3.6494575, -1.5500991, -3.6494575, -1.5500991, -0.9121119, 0.9122362
9: -2.9068365, -1.1850188, -2.9068365, -1.1850188, -0.7875482, 0.7875707

Time for backsubstitution: 6.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 361
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 651
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3525

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2605

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0929442, upper bound: 0.0929224
time: 8.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0929306, upper bound: 0.0929482
time: 6.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.2739444, -1.3582740, -3.2739444, -1.3582740, -1.5906918, 1.5907876
1: -3.1609457, -1.3756382, -3.1609457, -1.3756382, -1.2089503, 1.2089515
2: -1.0850873, -0.6183419, -1.0850873, -0.6183419, -0.2399458, 0.2399645
3: 0.3414853, 0.6887888, 0.3414853, 0.6887888, -0.2269136, 0.2269040
4: -4.2210059, -3.3748572, -4.2210059, -3.3748572, -0.2736188, 0.2736177
5: 1.8216184, 2.1860926, 1.8216184, 2.1860926, -0.1302797, 0.1302736
6: -4.4172969, -3.4960172, -4.4172969, -3.4960172, -0.1770852, 0.1770593
7: -0.4386549, 0.0872523, -0.4386549, 0.0872523, -0.1361896, 0.1361904
8: -3.6494575, -1.5500991, -3.6494575, -1.5500991, -0.9120461, 0.9123029
9: -2.9068365, -1.1850188, -2.9068365, -1.1850188, -0.7875347, 0.7875859

Time for backsubstitution: 6.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 361
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 651
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3525

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2605

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0929720, upper bound: 0.0928980
time: 7.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0929601, upper bound: 0.0929226
time: 188.33 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.2739444, -1.3582740, -3.2739444, -1.3582740, -1.5907879, 1.5906916
1: -3.1609457, -1.3756382, -3.1609457, -1.3756382, -1.2089515, 1.2089503
2: -1.0850873, -0.6183419, -1.0850873, -0.6183419, -0.2399645, 0.2399457
3: 0.3414853, 0.6887888, 0.3414853, 0.6887888, -0.2269040, 0.2269136
4: -4.2210059, -3.3748572, -4.2210059, -3.3748572, -0.2736177, 0.2736188
5: 1.8216184, 2.1860926, 1.8216184, 2.1860926, -0.1302736, 0.1302797
6: -4.4172969, -3.4960172, -4.4172969, -3.4960172, -0.1770594, 0.1770852
7: -0.4386549, 0.0872523, -0.4386549, 0.0872523, -0.1361904, 0.1361896
8: -3.6494575, -1.5500991, -3.6494575, -1.5500991, -0.9123029, 0.9120462
9: -2.9068365, -1.1850188, -2.9068365, -1.1850188, -0.7875859, 0.7875347

Time for backsubstitution: 6.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 361
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 651
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3525

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2605

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0929189, upper bound: 0.0929698
time: 10.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0928887, upper bound: 0.0929775
time: 64.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.2739444, -1.3582740, -3.2739444, -1.3582740, -1.5907617, 1.5907176
1: -3.1609457, -1.3756382, -3.1609457, -1.3756382, -1.2089543, 1.2089472
2: -1.0850873, -0.6183419, -1.0850873, -0.6183419, -0.2399604, 0.2399497
3: 0.3414853, 0.6887888, 0.3414853, 0.6887888, -0.2269052, 0.2269123
4: -4.2210059, -3.3748572, -4.2210059, -3.3748572, -0.2736204, 0.2736161
5: 1.8216184, 2.1860926, 1.8216184, 2.1860926, -0.1302733, 0.1302795
6: -4.4172969, -3.4960172, -4.4172969, -3.4960172, -0.1770600, 0.1770845
7: -0.4386549, 0.0872523, -0.4386549, 0.0872523, -0.1361905, 0.1361894
8: -3.6494575, -1.5500991, -3.6494575, -1.5500991, -0.9122361, 0.9121119
9: -2.9068365, -1.1850188, -2.9068365, -1.1850188, -0.7875707, 0.7875481

Time for backsubstitution: 6.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 361
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 651
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3525

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2605

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0929396, upper bound: 0.0929388
time: 105.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0929124, upper bound: 0.0929559
time: 11.13 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.2739444, -1.3582740, -3.2739444, -1.3582740, -1.5907433, 1.5907602
1: -3.1609457, -1.3756382, -3.1609457, -1.3756382, -1.2089536, 1.2089503
2: -1.0850873, -0.6183419, -1.0850873, -0.6183419, -0.2399659, 0.2399452
3: 0.3414853, 0.6887888, 0.3414853, 0.6887888, -0.2269062, 0.2269137
4: -4.2210059, -3.3748572, -4.2210059, -3.3748572, -0.2736191, 0.2736180
5: 1.8216184, 2.1860926, 1.8216184, 2.1860926, -0.1302735, 0.1302801
6: -4.4172969, -3.4960172, -4.4172969, -3.4960172, -0.1770720, 0.1770782
7: -0.4386549, 0.0872523, -0.4386549, 0.0872523, -0.1361907, 0.1361894
8: -3.6494575, -1.5500991, -3.6494575, -1.5500991, -0.9121120, 0.9122423
9: -2.9068365, -1.1850188, -2.9068365, -1.1850188, -0.7875890, 0.7875366

Time for backsubstitution: 6.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 361
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 651
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3525

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2605

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0929281, upper bound: 0.0929697
time: 7.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0929074, upper bound: 0.0929808
time: 9.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.2739444, -1.3582740, -3.2739444, -1.3582740, -1.5907168, 1.5907860
1: -3.1609457, -1.3756382, -3.1609457, -1.3756382, -1.2089560, 1.2089474
2: -1.0850873, -0.6183419, -1.0850873, -0.6183419, -0.2399618, 0.2399492
3: 0.3414853, 0.6887888, 0.3414853, 0.6887888, -0.2269073, 0.2269124
4: -4.2210059, -3.3748572, -4.2210059, -3.3748572, -0.2736217, 0.2736153
5: 1.8216184, 2.1860926, 1.8216184, 2.1860926, -0.1302732, 0.1302799
6: -4.4172969, -3.4960172, -4.4172969, -3.4960172, -0.1770726, 0.1770775
7: -0.4386549, 0.0872523, -0.4386549, 0.0872523, -0.1361908, 0.1361893
8: -3.6494575, -1.5500991, -3.6494575, -1.5500991, -0.9120445, 0.9123080
9: -2.9068365, -1.1850188, -2.9068365, -1.1850188, -0.7875725, 0.7875497

Time for backsubstitution: 6.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 361
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 651
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3525

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2605

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0929493, upper bound: 0.0928721
time: 423.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0929324, upper bound: 0.0928914
time: 99.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.2739444, -1.3582740, -3.2739444, -1.3582740, -1.5907173, 1.5907612
1: -3.1609457, -1.3756382, -3.1609457, -1.3756382, -1.2089458, 1.2089555
2: -1.0850873, -0.6183419, -1.0850873, -0.6183419, -0.2399502, 0.2399600
3: 0.3414853, 0.6887888, 0.3414853, 0.6887888, -0.2269123, 0.2269052
4: -4.2210059, -3.3748572, -4.2210059, -3.3748572, -0.2736177, 0.2736188
5: 1.8216184, 2.1860926, 1.8216184, 2.1860926, -0.1302795, 0.1302732
6: -4.4172969, -3.4960172, -4.4172969, -3.4960172, -0.1770855, 0.1770590
7: -0.4386549, 0.0872523, -0.4386549, 0.0872523, -0.1361893, 0.1361906
8: -3.6494575, -1.5500991, -3.6494575, -1.5500991, -0.9121646, 0.9121827
9: -2.9068365, -1.1850188, -2.9068365, -1.1850188, -0.7875439, 0.7875744

Time for backsubstitution: 6.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 361
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 651
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3525

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2605

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0929746, upper bound: 0.0928966
time: 152.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0929547, upper bound: 0.0929201
time: 8.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.2739444, -1.3582740, -3.2739444, -1.3582740, -1.5906911, 1.5907872
1: -3.1609457, -1.3756382, -3.1609457, -1.3756382, -1.2089486, 1.2089527
2: -1.0850873, -0.6183419, -1.0850873, -0.6183419, -0.2399461, 0.2399640
3: 0.3414853, 0.6887888, 0.3414853, 0.6887888, -0.2269136, 0.2269039
4: -4.2210059, -3.3748572, -4.2210059, -3.3748572, -0.2736204, 0.2736161
5: 1.8216184, 2.1860926, 1.8216184, 2.1860926, -0.1302796, 0.1302734
6: -4.4172969, -3.4960172, -4.4172969, -3.4960172, -0.1770861, 0.1770583
7: -0.4386549, 0.0872523, -0.4386549, 0.0872523, -0.1361894, 0.1361905
8: -3.6494575, -1.5500991, -3.6494575, -1.5500991, -0.9120977, 0.9122484
9: -2.9068365, -1.1850188, -2.9068365, -1.1850188, -0.7875299, 0.7875890

Time for backsubstitution: 6.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 394
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 379
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 728
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 361
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 651
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 1110
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2278
type: DSZ, layer: 1, pos: 2279
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2581
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2680
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3525

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2605

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0929980, upper bound: 0.0928742
time: 212.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0929827, upper bound: 0.0928975
time: 7.76 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 227.01 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 227.01
Output dim: 5, lower bound: -0.0928885, upper bound: 0.0930079
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 227.01
Output dim: 5, lower bound: -0.0928668, upper bound: 0.0930169
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 227.01
Output dim: 5, lower bound: -0.0929110, upper bound: 0.0929792
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 227.01
Output dim: 5, lower bound: -0.0928932, upper bound: 0.0929916
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 227.01
Output dim: 5, lower bound: -0.0928882, upper bound: 0.0929844
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 227.01
Output dim: 5, lower bound: -0.0928681, upper bound: 0.0929143
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 227.01
Output dim: 5, lower bound: -0.0929111, upper bound: 0.0929649
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 227.01
Output dim: 5, lower bound: -0.0928931, upper bound: 0.0929834
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 227.01
Output dim: 5, lower bound: -0.0929447, upper bound: 0.0929353
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 227.01
Output dim: 5, lower bound: -0.0929318, upper bound: 0.0929580
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 227.01
Output dim: 5, lower bound: -0.0929716, upper bound: 0.0929160
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 227.01
Output dim: 5, lower bound: -0.0929586, upper bound: 0.0929358
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 227.01
Output dim: 5, lower bound: -0.0929442, upper bound: 0.0929224
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 227.01
Output dim: 5, lower bound: -0.0929306, upper bound: 0.0929482
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 227.01
Output dim: 5, lower bound: -0.0929720, upper bound: 0.0928980
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 227.01
Output dim: 5, lower bound: -0.0929601, upper bound: 0.0929226
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 227.01
Output dim: 5, lower bound: -0.0929189, upper bound: 0.0929698
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 227.01
Output dim: 5, lower bound: -0.0928887, upper bound: 0.0929775
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 227.01
Output dim: 5, lower bound: -0.0929396, upper bound: 0.0929388
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 227.01
Output dim: 5, lower bound: -0.0929124, upper bound: 0.0929559
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 227.01
Output dim: 5, lower bound: -0.0929281, upper bound: 0.0929697
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 227.01
Output dim: 5, lower bound: -0.0929074, upper bound: 0.0929808
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 227.01
Output dim: 5, lower bound: -0.0929493, upper bound: 0.0928721
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 227.01
Output dim: 5, lower bound: -0.0929324, upper bound: 0.0928914
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 227.01
Output dim: 5, lower bound: -0.0929746, upper bound: 0.0928966
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 227.01
Output dim: 5, lower bound: -0.0929547, upper bound: 0.0929201
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 227.01
Output dim: 5, lower bound: -0.0929980, upper bound: 0.0928742
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 227.01
Output dim: 5, lower bound: -0.0929827, upper bound: 0.0928975
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 227.01
Output dim: 5, lower bound: -0.0930247, upper bound: 0.0929540
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 227.01
Output dim: 5, lower bound: -0.0930510, upper bound: 0.0929273

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 85.14 + 3567.99 = 3653.13 seconds
