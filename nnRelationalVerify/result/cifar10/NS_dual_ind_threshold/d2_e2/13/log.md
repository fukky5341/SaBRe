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
execution time: IAR + RelationalAnalysis = 7.93 + 77.28 = 85.21 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0930953, upper bound: 0.0931023

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 361
type: A, layer: 1, pos: 394
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 2139
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 2426
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 379
type: A, layer: 1, pos: 2170
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 3125
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2436
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 3010
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 2507
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 2972
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 3538
type: A, layer: 1, pos: 3127
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 2680
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3351
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 2187
type: A, layer: 1, pos: 2085
type: A, layer: 1, pos: 2317
type: A, layer: 1, pos: 2070
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 2657
type: A, layer: 1, pos: 2451
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2550
type: A, layer: 1, pos: 2179
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 423
type: A, layer: 1, pos: 2213
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 2676
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 2331
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2209
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 3525
type: A, layer: 1, pos: 2408
type: A, layer: 1, pos: 2282
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 2672
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 2400
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 2956
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 2435
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2116
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 2595
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 3114
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2671
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 2322
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2997
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2037
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 3253
type: A, layer: 1, pos: 2250
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 2278
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 2926
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 3290
type: A, layer: 1, pos: 2081
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 2082
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 2925
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 3514
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 2928
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 2927
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 438
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2084
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2244
type: A, layer: 1, pos: 2245
type: A, layer: 1, pos: 2246
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 2685
type: A, layer: 1, pos: 2686
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 3149
type: A, layer: 1, pos: 3368

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 361

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0928245, upper bound: 0.0930947
time: 56.18 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930938, upper bound: 0.0930968
time: 231.25 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 287.49 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 287.49
Output dim: 5, lower bound: -0.0928245, upper bound: 0.0930947
NS_A2, status: Status.UNKNOWN, split count: 1, time: 287.49
Output dim: 5, lower bound: -0.0930938, upper bound: 0.0930968

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -3.2715757, -1.3621938, -3.2734339, -1.3613925, -1.5878007, 1.5892227
1: -3.1605997, -1.3764551, -3.1609373, -1.3762863, -1.2110991, 1.2115209
2: -1.0838552, -0.6202608, -1.0846615, -0.6198844, -0.2378764, 0.2382519
3: 0.3436911, 0.6881120, 0.3432645, 0.6887383, -0.2251511, 0.2251795
4: -4.2184510, -3.3814681, -4.2209897, -3.3801425, -0.2663270, 0.2674566
5: 1.8252333, 2.1845481, 1.8244948, 2.1860771, -0.1272420, 0.1265418
6: -4.4164839, -3.4963896, -4.4166484, -3.4960287, -0.1774336, 0.1772762
7: -0.4386553, 0.0872433, -0.4386515, 0.0872452, -0.1360831, 0.1360750
8: -3.6492808, -1.5493121, -3.6493154, -1.5502305, -0.9175094, 0.9184458
9: -2.9068131, -1.1849866, -2.9068177, -1.1850595, -0.7922242, 0.7928925

Time for backsubstitution: 6.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 394
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 2139
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 2605
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2426
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 2585
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 361
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 379
type: B, layer: 1, pos: 2170
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 3125
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2436
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 3010
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 2091
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 2507
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 2972
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 3538
type: B, layer: 1, pos: 3127
type: B, layer: 1, pos: 2231
type: B, layer: 1, pos: 2680
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3351
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 2187
type: B, layer: 1, pos: 2085
type: B, layer: 1, pos: 2317
type: B, layer: 1, pos: 2070
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 2657
type: B, layer: 1, pos: 2451
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2550
type: B, layer: 1, pos: 2179
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 423
type: B, layer: 1, pos: 2213
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 2203
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 2331
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2209
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 3525
type: B, layer: 1, pos: 2408
type: B, layer: 1, pos: 2282
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 2672
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 2400
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 2956
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 2435
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2116
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 2595
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 3114
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 2671
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 2322
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2997
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 2037
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 3253
type: B, layer: 1, pos: 2250
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2278
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 2926
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 3290
type: B, layer: 1, pos: 2081
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 2082
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 2925
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 2030
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 3514
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 2928
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 2927
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 438
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2084
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2244
type: B, layer: 1, pos: 2245
type: B, layer: 1, pos: 2246
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 2686
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 3149
type: B, layer: 1, pos: 3368

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 394

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0926483, upper bound: 0.0930778
time: 10.35 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0928084, upper bound: 0.0930812
time: 22.16 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -3.2738869, -1.3582840, -3.2739000, -1.3582816, -1.5933728, 1.5877793
1: -3.1609440, -1.3756404, -3.1609442, -1.3756397, -1.2122349, 1.2109594
2: -1.0850536, -0.6183699, -1.0850611, -0.6183680, -0.2401507, 0.2389307
3: 0.3415635, 0.6887829, 0.3415461, 0.6887842, -0.2252718, 0.2270795
4: -4.2210054, -3.3748803, -4.2210059, -3.3748751, -0.2737077, 0.2659504
5: 1.8216257, 2.1860907, 1.8216240, 2.1860909, -0.1261194, 0.1308557
6: -4.4172940, -3.4960184, -4.4172945, -3.4960179, -0.1772917, 0.1780225
7: -0.4386539, 0.0872522, -0.4386544, 0.0872521, -0.1360874, 0.1360926
8: -3.6489713, -1.5501008, -3.6490798, -1.5501003, -0.9185010, 0.9180543
9: -2.9067779, -1.1850190, -2.9067912, -1.1850190, -0.7934840, 0.7925789

Time for backsubstitution: 6.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 394
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 2139
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 2605
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2426
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 361
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 2585
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 379
type: B, layer: 1, pos: 2170
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 3125
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2436
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 3010
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 2091
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 2507
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 2972
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 3538
type: B, layer: 1, pos: 3127
type: B, layer: 1, pos: 2231
type: B, layer: 1, pos: 2680
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3351
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 2187
type: B, layer: 1, pos: 2085
type: B, layer: 1, pos: 2317
type: B, layer: 1, pos: 2070
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 2657
type: B, layer: 1, pos: 2451
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2550
type: B, layer: 1, pos: 2179
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 423
type: B, layer: 1, pos: 2213
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 2203
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 2331
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2209
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 3525
type: B, layer: 1, pos: 2408
type: B, layer: 1, pos: 2282
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 2672
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 2400
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 2956
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 2435
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2116
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 2595
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 3114
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 2671
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 2322
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2997
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 2037
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 3253
type: B, layer: 1, pos: 2250
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2278
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 2926
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 3290
type: B, layer: 1, pos: 2081
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 2082
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 2925
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 2030
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 3514
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 2928
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 2927
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 438
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2084
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2244
type: B, layer: 1, pos: 2245
type: B, layer: 1, pos: 2246
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 2686
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 3149
type: B, layer: 1, pos: 3368

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 394

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0929188, upper bound: 0.0930811
time: 35.05 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930781, upper bound: 0.0930842
time: 46.77 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 88.26 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 88.26
Output dim: 5, lower bound: -0.0926483, upper bound: 0.0930778
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 88.26
Output dim: 5, lower bound: -0.0928084, upper bound: 0.0930812
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 88.26
Output dim: 5, lower bound: -0.0929188, upper bound: 0.0930811
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 88.26
Output dim: 5, lower bound: -0.0930781, upper bound: 0.0930842

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -3.2714732, -1.3621953, -3.2732949, -1.3613942, -1.5876037, 1.5889890
1: -3.1605723, -1.3764563, -3.1608994, -1.3762882, -1.2107229, 1.2110305
2: -1.0827713, -0.6202711, -1.0831677, -0.6198978, -0.2367385, 0.2366751
3: 0.3436913, 0.6881057, 0.3432649, 0.6887310, -0.2249966, 0.2250573
4: -4.2182155, -3.3814819, -4.2206712, -3.3801615, -0.2660422, 0.2670905
5: 1.8252345, 2.1840165, 1.8244966, 2.1854119, -0.1264607, 0.1259247
6: -4.4164543, -3.4963906, -4.4166074, -3.4960306, -0.1773854, 0.1772170
7: -0.4381989, 0.0872414, -0.4380109, 0.0872426, -0.1356203, 0.1354285
8: -3.6492629, -1.5501812, -3.6492922, -1.5514505, -0.9162469, 0.9175297
9: -2.9067852, -1.1849949, -2.9067793, -1.1850708, -0.7916726, 0.7921691

Time for backsubstitution: 6.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 2139
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 2426
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 394
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 379
type: A, layer: 1, pos: 2170
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 3125
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2436
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 3010
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 2507
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 2972
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 3538
type: A, layer: 1, pos: 3127
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 2680
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3351
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 2187
type: A, layer: 1, pos: 2085
type: A, layer: 1, pos: 2317
type: A, layer: 1, pos: 2070
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 2657
type: A, layer: 1, pos: 2451
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2550
type: A, layer: 1, pos: 2179
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 423
type: A, layer: 1, pos: 2213
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 2676
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 2331
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2209
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 3525
type: A, layer: 1, pos: 2408
type: A, layer: 1, pos: 2282
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 2672
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 2400
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 2956
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 2435
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2116
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 2595
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 3114
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2671
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 2322
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2997
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2037
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 3253
type: A, layer: 1, pos: 2250
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 2278
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 2926
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 3290
type: A, layer: 1, pos: 2081
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 2082
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 2925
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 3514
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 2928
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 2927
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 438
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2084
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2244
type: A, layer: 1, pos: 2245
type: A, layer: 1, pos: 2246
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 2685
type: A, layer: 1, pos: 2686
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 3149
type: A, layer: 1, pos: 3368

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3064

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0925288, upper bound: 0.0930277
time: 309.00 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0926255, upper bound: 0.0930519
time: 28.93 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -3.2691841, -1.3621938, -3.2705817, -1.3621030, -1.5859175, 1.5886180
1: -3.1561539, -1.3764551, -3.1554763, -1.3771074, -1.2088717, 1.2108762
2: -1.0838184, -0.6202643, -1.0848241, -0.6143078, -0.2430900, 0.2372798
3: 0.3436925, 0.6873320, 0.3434275, 0.6878135, -0.2248548, 0.2254447
4: -4.2184081, -3.3814683, -4.2210078, -3.3795838, -0.2667186, 0.2673231
5: 1.8252356, 2.1843514, 1.8218329, 2.1858580, -0.1266430, 0.1288081
6: -4.4164829, -3.4963906, -4.4166489, -3.4958255, -0.1774644, 0.1772424
7: -0.4386215, 0.0872432, -0.4385968, 0.0896221, -0.1384222, 0.1355638
8: -3.6492741, -1.5493603, -3.6536300, -1.5502398, -0.9166296, 0.9220251
9: -2.9022634, -1.1849899, -2.9008915, -1.1861989, -0.7910591, 0.7920699

Time for backsubstitution: 6.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 2139
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 2426
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 394
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 379
type: A, layer: 1, pos: 2170
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 3125
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2436
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 3010
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 2507
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 2972
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 3538
type: A, layer: 1, pos: 3127
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 2680
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3351
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 2187
type: A, layer: 1, pos: 2085
type: A, layer: 1, pos: 2317
type: A, layer: 1, pos: 2070
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 2657
type: A, layer: 1, pos: 2451
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2550
type: A, layer: 1, pos: 2179
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 423
type: A, layer: 1, pos: 2213
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 2676
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 2331
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2209
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 3525
type: A, layer: 1, pos: 2408
type: A, layer: 1, pos: 2282
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 2672
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 2400
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 2956
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 2435
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2116
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 2595
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 3114
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2671
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 2322
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2997
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2037
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 3253
type: A, layer: 1, pos: 2250
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 2278
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 2926
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 3290
type: A, layer: 1, pos: 2081
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 2082
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 2925
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 3514
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 2928
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 2927
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 438
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2084
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2244
type: A, layer: 1, pos: 2245
type: A, layer: 1, pos: 2246
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 2685
type: A, layer: 1, pos: 2686
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 3149
type: A, layer: 1, pos: 3368

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3064

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0926888, upper bound: 0.0930301
time: 163.18 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0927856, upper bound: 0.0930615
time: 5.67 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -3.2737842, -1.3582859, -3.2737617, -1.3582828, -1.5931752, 1.5875444
1: -3.1609173, -1.3756411, -3.1609063, -1.3756411, -1.2118590, 1.2104700
2: -1.0839701, -0.6183804, -1.0835671, -0.6183811, -0.2390129, 0.2373531
3: 0.3415637, 0.6887767, 0.3415463, 0.6887768, -0.2251171, 0.2269570
4: -4.2207704, -3.3748944, -4.2206874, -3.3748946, -0.2734231, 0.2655841
5: 1.8216271, 2.1855588, 1.8216259, 2.1854258, -0.1253380, 0.1302387
6: -4.4172649, -3.4960189, -4.4172530, -3.4960194, -0.1772435, 0.1779633
7: -0.4381974, 0.0872501, -0.4380139, 0.0872493, -0.1356246, 0.1354463
8: -3.6489527, -1.5509703, -3.6490560, -1.5513198, -0.9172385, 0.9171382
9: -2.9067502, -1.1850278, -2.9067521, -1.1850297, -0.7929327, 0.7918558

Time for backsubstitution: 6.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 2139
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 2426
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 394
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 379
type: A, layer: 1, pos: 2170
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 3125
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2436
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 3010
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 2507
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 2972
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 3538
type: A, layer: 1, pos: 3127
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 2680
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3351
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 2187
type: A, layer: 1, pos: 2085
type: A, layer: 1, pos: 2317
type: A, layer: 1, pos: 2070
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 2657
type: A, layer: 1, pos: 2451
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2550
type: A, layer: 1, pos: 2179
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 423
type: A, layer: 1, pos: 2213
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 2676
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 2331
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2209
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 3525
type: A, layer: 1, pos: 2408
type: A, layer: 1, pos: 2282
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 2672
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 2400
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 2956
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 2435
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2116
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 2595
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 3114
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2671
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 2322
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2997
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2037
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 3253
type: A, layer: 1, pos: 2250
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 2278
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 2926
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 3290
type: A, layer: 1, pos: 2081
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 2082
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 2925
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 3514
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 2928
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 2927
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 438
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2084
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2244
type: A, layer: 1, pos: 2245
type: A, layer: 1, pos: 2246
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 2685
type: A, layer: 1, pos: 2686
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 3149
type: A, layer: 1, pos: 3368

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3064

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0927982, upper bound: 0.0930304
time: 141.81 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0928946, upper bound: 0.0930608
time: 14.50 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -3.2714944, -1.3582842, -3.2710502, -1.3589916, -1.5914898, 1.5871749
1: -3.1564987, -1.3756409, -3.1554840, -1.3764603, -1.2100070, 1.2103150
2: -1.0850159, -0.6183733, -1.0852240, -0.6127913, -0.2453638, 0.2379580
3: 0.3415649, 0.6880031, 0.3417091, 0.6878597, -0.2249753, 0.2273452
4: -4.2209630, -3.3748806, -4.2210250, -3.3743176, -0.2740989, 0.2658158
5: 1.8216281, 2.1858935, 1.8189625, 2.1858718, -0.1255203, 0.1331220
6: -4.4172935, -3.4960194, -4.4172950, -3.4958138, -0.1773223, 0.1779886
7: -0.4386201, 0.0872521, -0.4385994, 0.0896291, -0.1384265, 0.1355815
8: -3.6489639, -1.5501492, -3.6533940, -1.5501094, -0.9176219, 0.9216334
9: -2.9022281, -1.1850212, -2.9008651, -1.1861575, -0.7923187, 0.7917564

Time for backsubstitution: 6.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 2139
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 2426
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 394
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 379
type: A, layer: 1, pos: 2170
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 3125
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2436
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 3010
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 2507
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 2972
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 3538
type: A, layer: 1, pos: 3127
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 2680
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3351
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 2187
type: A, layer: 1, pos: 2085
type: A, layer: 1, pos: 2317
type: A, layer: 1, pos: 2070
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 2657
type: A, layer: 1, pos: 2451
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2550
type: A, layer: 1, pos: 2179
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 423
type: A, layer: 1, pos: 2213
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 2676
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 2331
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2209
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 3525
type: A, layer: 1, pos: 2408
type: A, layer: 1, pos: 2282
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 2672
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 2400
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 2956
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 2435
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2116
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 2595
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 3114
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2671
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 2322
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2997
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2037
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 3253
type: A, layer: 1, pos: 2250
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 2278
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 2926
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 3290
type: A, layer: 1, pos: 2081
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 2082
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 2925
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 3514
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 2928
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 2927
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 438
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2084
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2244
type: A, layer: 1, pos: 2245
type: A, layer: 1, pos: 2246
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 2685
type: A, layer: 1, pos: 2686
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 3149
type: A, layer: 1, pos: 3368

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3064

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0929588, upper bound: 0.0930356
time: 11.79 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930536, upper bound: 0.0930618
time: 17.79 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 36.05 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 36.05
Output dim: 5, lower bound: -0.0925288, upper bound: 0.0930277
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 36.05
Output dim: 5, lower bound: -0.0926255, upper bound: 0.0930519
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 36.05
Output dim: 5, lower bound: -0.0926888, upper bound: 0.0930301
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 36.05
Output dim: 5, lower bound: -0.0927856, upper bound: 0.0930615
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 36.05
Output dim: 5, lower bound: -0.0927982, upper bound: 0.0930304
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 36.05
Output dim: 5, lower bound: -0.0928946, upper bound: 0.0930608
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 36.05
Output dim: 5, lower bound: -0.0929588, upper bound: 0.0930356
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 36.05
Output dim: 5, lower bound: -0.0930536, upper bound: 0.0930618

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -3.2698119, -1.3709919, -3.2730627, -1.3689566, -1.5760634, 1.5790756
1: -3.1581185, -1.3870127, -3.1607642, -1.3850486, -1.1933081, 1.1977839
2: -1.0825926, -0.6208189, -1.0830121, -0.6203415, -0.2357646, 0.2358861
3: 0.3474070, 0.6884240, 0.3464042, 0.6886958, -0.2205077, 0.2212216
4: -4.2170320, -3.3810685, -4.2196808, -3.3801694, -0.2642523, 0.2653464
5: 1.8296157, 2.1839421, 1.8281752, 2.1854048, -0.1212194, 0.1209376
6: -4.4161901, -3.4961939, -4.4163828, -3.4960475, -0.1768818, 0.1766187
7: -0.4341379, 0.0873602, -0.4345832, 0.0872282, -0.1304512, 0.1308252
8: -3.6477823, -1.5570936, -3.6491973, -1.5574193, -0.9011181, 0.9067222
9: -2.9044013, -1.1936970, -2.9067426, -1.1922877, -0.7727435, 0.7785995

Time for backsubstitution: 6.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 2139
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2605
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2426
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 361
type: B, layer: 1, pos: 2585
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 379
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 2170
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 3125
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2436
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 3010
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 2091
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 2507
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2972
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 3538
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 2231
type: B, layer: 1, pos: 3127
type: B, layer: 1, pos: 2680
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3351
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2085
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 2187
type: B, layer: 1, pos: 2317
type: B, layer: 1, pos: 2070
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 2657
type: B, layer: 1, pos: 2451
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2550
type: B, layer: 1, pos: 2179
type: B, layer: 1, pos: 423
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 2213
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 2203
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 2331
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 2209
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 3525
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 2282
type: B, layer: 1, pos: 2408
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 2672
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 2400
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 2956
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 2435
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 2116
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 2595
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 3114
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 2671
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 2322
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 2997
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 2037
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 2250
type: B, layer: 1, pos: 3253
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2278
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 2926
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 3290
type: B, layer: 1, pos: 2081
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 2082
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 2925
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 2030
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 3514
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 2928
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 2927
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 438
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2084
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2244
type: B, layer: 1, pos: 2245
type: B, layer: 1, pos: 2246
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 2686
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 3149
type: B, layer: 1, pos: 3368

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2377

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0924985, upper bound: 0.0929428
time: 300.86 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0925046, upper bound: 0.0930022
time: 21.42 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -3.2714117, -1.3632324, -3.2732413, -1.3624065, -1.5871091, 1.5835838
1: -3.1605535, -1.3774977, -3.1608803, -1.3773012, -1.2106007, 1.1976497
2: -1.0827129, -0.6204235, -1.0831175, -0.6200306, -0.2366628, 0.2360558
3: 0.3442498, 0.6880918, 0.3437645, 0.6887189, -0.2240120, 0.2247470
4: -4.2178006, -3.3814831, -4.2203255, -3.3801634, -0.2648137, 0.2668849
5: 1.8256238, 2.1840136, 1.8248734, 2.1854091, -0.1247540, 0.1256147
6: -4.4162741, -3.4963984, -4.4164581, -3.4960375, -0.1767724, 0.1771805
7: -0.4377500, 0.0872394, -0.4375847, 0.0872406, -0.1345016, 0.1350962
8: -3.6492488, -1.5506220, -3.6492791, -1.5518734, -0.9161664, 0.9034328
9: -2.9067805, -1.1856303, -2.9067738, -1.1856787, -0.7915680, 0.7723186

Time for backsubstitution: 6.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 2139
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 2605
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2426
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 2585
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 361
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 379
type: B, layer: 1, pos: 2170
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 3125
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2436
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 3010
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 2091
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 2507
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 2972
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 3538
type: B, layer: 1, pos: 2231
type: B, layer: 1, pos: 3127
type: B, layer: 1, pos: 2680
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3351
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 2187
type: B, layer: 1, pos: 2085
type: B, layer: 1, pos: 2317
type: B, layer: 1, pos: 2070
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 2657
type: B, layer: 1, pos: 2451
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2550
type: B, layer: 1, pos: 2179
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 423
type: B, layer: 1, pos: 2213
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 2203
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 2331
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2209
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 3525
type: B, layer: 1, pos: 2408
type: B, layer: 1, pos: 2282
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 2672
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 2400
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 2956
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 2435
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2116
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 2595
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 3114
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 2671
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 2322
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2997
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 2037
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 3253
type: B, layer: 1, pos: 2250
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2278
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 2926
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 3290
type: B, layer: 1, pos: 2081
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 2082
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 2925
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 2030
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 3514
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 2928
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 2927
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 438
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2084
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2244
type: B, layer: 1, pos: 2245
type: B, layer: 1, pos: 2246
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 2686
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 3149
type: B, layer: 1, pos: 3368

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2377

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0925963, upper bound: 0.0929735
time: 7.50 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0926005, upper bound: 0.0930345
time: 5.70 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3.2675221, -1.3709908, -3.2703488, -1.3696656, -1.5743697, 1.5787046
1: -3.1536973, -1.3870111, -3.1553371, -1.3858674, -1.1914518, 1.1976280
2: -1.0836391, -0.6208119, -1.0846679, -0.6147517, -0.2421157, 0.2364916
3: 0.3474081, 0.6876504, 0.3465583, 0.6877786, -0.2203659, 0.2216212
4: -4.2172241, -3.3810554, -4.2200184, -3.3795919, -0.2649289, 0.2655788
5: 1.8296167, 2.1842768, 1.8254993, 2.1858511, -0.1214018, 0.1238591
6: -4.4162188, -3.4961951, -4.4164238, -3.4958420, -0.1769606, 0.1766441
7: -0.4345611, 0.0873621, -0.4351684, 0.0896082, -0.1332532, 0.1309602
8: -3.6477947, -1.5562723, -3.6535349, -1.5562105, -0.9015008, 0.9112178
9: -2.8998797, -1.1936915, -2.9008551, -1.1934154, -0.7721292, 0.7785003

Time for backsubstitution: 6.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 2139
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2605
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2426
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 361
type: B, layer: 1, pos: 2585
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 379
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 2170
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 3125
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2436
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 3010
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 2091
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 2507
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2972
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 3538
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 2231
type: B, layer: 1, pos: 3127
type: B, layer: 1, pos: 2680
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3351
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2085
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 2187
type: B, layer: 1, pos: 2317
type: B, layer: 1, pos: 2070
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 2657
type: B, layer: 1, pos: 2451
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2550
type: B, layer: 1, pos: 2179
type: B, layer: 1, pos: 423
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 2213
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 2203
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 2331
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 2209
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 3525
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 2282
type: B, layer: 1, pos: 2408
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 2672
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 2400
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 2956
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 2435
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 2116
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 2595
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 3114
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 2671
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 2322
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 2997
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 2037
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 2250
type: B, layer: 1, pos: 3253
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2278
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 2926
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 3290
type: B, layer: 1, pos: 2081
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 2082
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 2925
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 2030
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 3514
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 2928
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 2927
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 438
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2084
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2244
type: B, layer: 1, pos: 2245
type: B, layer: 1, pos: 2246
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 2686
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 3149
type: B, layer: 1, pos: 3368

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2377

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0926587, upper bound: 0.0929439
time: 13.53 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0926624, upper bound: 0.0929458
time: 160.06 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3.2691221, -1.3632309, -3.2705286, -1.3631153, -1.5854222, 1.5832129
1: -3.1561348, -1.3774962, -3.1554570, -1.3781195, -1.2087491, 1.1974945
2: -1.0837600, -0.6204165, -1.0847738, -0.6144408, -0.2430144, 0.2366607
3: 0.3442510, 0.6873181, 0.3439265, 0.6878017, -0.2238702, 0.2251353
4: -4.2179928, -3.3814695, -4.2206626, -3.3795853, -0.2654902, 0.2671177
5: 1.8256242, 2.1843486, 1.8222086, 2.1858554, -0.1249363, 0.1285015
6: -4.4163036, -3.4963994, -4.4165010, -3.4958320, -0.1768515, 0.1772058
7: -0.4381730, 0.0872411, -0.4381706, 0.0896202, -0.1373036, 0.1352313
8: -3.6492610, -1.5498013, -3.6536174, -1.5506623, -0.9165496, 0.9079283
9: -2.9022589, -1.1856251, -2.9008865, -1.1868076, -0.7909543, 0.7722195

Time for backsubstitution: 6.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 2139
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 2605
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2426
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 2585
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 361
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 379
type: B, layer: 1, pos: 2170
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 3125
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2436
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 3010
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 2091
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 2507
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 2972
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 3538
type: B, layer: 1, pos: 2231
type: B, layer: 1, pos: 3127
type: B, layer: 1, pos: 2680
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3351
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 2187
type: B, layer: 1, pos: 2085
type: B, layer: 1, pos: 2317
type: B, layer: 1, pos: 2070
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 2657
type: B, layer: 1, pos: 2451
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2550
type: B, layer: 1, pos: 2179
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 423
type: B, layer: 1, pos: 2213
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 2203
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 2331
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2209
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 3525
type: B, layer: 1, pos: 2408
type: B, layer: 1, pos: 2282
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 2672
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 2400
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 2956
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 2435
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2116
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 2595
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 3114
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 2671
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 2322
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2997
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 2037
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 3253
type: B, layer: 1, pos: 2250
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2278
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 2926
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 3290
type: B, layer: 1, pos: 2081
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 2082
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 2925
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 2030
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 3514
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 2928
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 2927
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 438
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2084
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2244
type: B, layer: 1, pos: 2245
type: B, layer: 1, pos: 2246
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 2686
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 3149
type: B, layer: 1, pos: 3368

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2377

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0927561, upper bound: 0.0929774
time: 7.07 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0927603, upper bound: 0.0930356
time: 5.41 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -3.2721214, -1.3670814, -3.2735293, -1.3658454, -1.5816321, 1.5776300
1: -3.1584609, -1.3861983, -3.1607711, -1.3844028, -1.1944433, 1.1972225
2: -1.0837913, -0.6189281, -1.0834115, -0.6188250, -0.2380390, 0.2365643
3: 0.3452869, 0.6890946, 0.3446921, 0.6887418, -0.2206284, 0.2231119
4: -4.2195864, -3.3744810, -4.2196980, -3.3749022, -0.2716366, 0.2638366
5: 1.8260081, 2.1854844, 1.8253045, 2.1854186, -0.1200964, 0.1252517
6: -4.4169998, -3.4958229, -4.4170289, -3.4960368, -0.1767401, 0.1773648
7: -0.4341363, 0.0873690, -0.4345860, 0.0872352, -0.1304557, 0.1308429
8: -3.6474729, -1.5578821, -3.6489601, -1.5572891, -0.9021099, 0.9063314
9: -2.9043658, -1.1937289, -2.9067166, -1.1922481, -0.7740034, 0.7782861

Time for backsubstitution: 6.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 2139
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2605
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2426
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 361
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 2585
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 379
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 2170
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 3125
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2436
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 3010
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 2091
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 2507
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2972
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 3538
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 2231
type: B, layer: 1, pos: 3127
type: B, layer: 1, pos: 2680
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3351
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2085
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 2187
type: B, layer: 1, pos: 2317
type: B, layer: 1, pos: 2070
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 2657
type: B, layer: 1, pos: 2451
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2550
type: B, layer: 1, pos: 2179
type: B, layer: 1, pos: 423
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 2213
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 2203
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 2331
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 2209
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 3525
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 2282
type: B, layer: 1, pos: 2408
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 2672
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 2400
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 2956
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 2435
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 2116
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 2595
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 3114
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 2671
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 2322
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 2997
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 2037
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 2250
type: B, layer: 1, pos: 3253
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2278
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 2926
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 3290
type: B, layer: 1, pos: 2081
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 2082
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 2925
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 2030
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 3514
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 2928
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 2927
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 438
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2084
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2244
type: B, layer: 1, pos: 2245
type: B, layer: 1, pos: 2246
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 2686
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 3149
type: B, layer: 1, pos: 3368

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2377

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0927708, upper bound: 0.0929465
time: 147.17 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0927747, upper bound: 0.0930109
time: 12.59 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -3.2737231, -1.3593218, -3.2737079, -1.3592961, -1.5926809, 1.5821395
1: -3.1608980, -1.3766828, -3.1608880, -1.3766553, -1.2117382, 1.1970890
2: -1.0839117, -0.6185327, -1.0835171, -0.6185141, -0.2389375, 0.2367337
3: 0.3421228, 0.6887627, 0.3420466, 0.6887649, -0.2241322, 0.2266458
4: -4.2203550, -3.3748956, -4.2203412, -3.3748960, -0.2721945, 0.2653786
5: 1.8220160, 2.1855559, 1.8220028, 2.1854231, -0.1236315, 0.1299286
6: -4.4170847, -3.4960275, -4.4171052, -3.4960263, -0.1766306, 0.1779268
7: -0.4377489, 0.0872480, -0.4375874, 0.0872474, -0.1345060, 0.1351139
8: -3.6489398, -1.5514116, -3.6490426, -1.5517428, -0.9171582, 0.9030414
9: -2.9067450, -1.1856623, -2.9067466, -1.1856380, -0.7928280, 0.7720060

Time for backsubstitution: 6.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 2139
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 2605
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2426
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 361
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 2585
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 379
type: B, layer: 1, pos: 2170
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 3125
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2436
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 3010
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 2091
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 2507
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 2972
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 3538
type: B, layer: 1, pos: 2231
type: B, layer: 1, pos: 3127
type: B, layer: 1, pos: 2680
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3351
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 2187
type: B, layer: 1, pos: 2085
type: B, layer: 1, pos: 2317
type: B, layer: 1, pos: 2070
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 2657
type: B, layer: 1, pos: 2451
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2550
type: B, layer: 1, pos: 2179
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 423
type: B, layer: 1, pos: 2213
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 2203
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 2331
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2209
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 3525
type: B, layer: 1, pos: 2408
type: B, layer: 1, pos: 2282
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 2672
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 2400
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 2956
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 2435
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2116
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 2595
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 3114
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 2671
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 2322
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2997
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 2037
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 3253
type: B, layer: 1, pos: 2250
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2278
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 2926
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 3290
type: B, layer: 1, pos: 2081
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 2082
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 2925
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 2030
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 3514
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 2928
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 2927
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 438
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2084
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2244
type: B, layer: 1, pos: 2245
type: B, layer: 1, pos: 2246
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 2686
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 3149
type: B, layer: 1, pos: 3368

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2377

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0928674, upper bound: 0.0929789
time: 9.31 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0928713, upper bound: 0.0930336
time: 23.64 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3.2698307, -1.3670807, -3.2708147, -1.3665538, -1.5799391, 1.5772593
1: -3.1540408, -1.3861964, -3.1553445, -1.3852210, -1.1925876, 1.1970673
2: -1.0848362, -0.6189210, -1.0850676, -0.6132352, -0.2443891, 0.2371697
3: 0.3452881, 0.6883211, 0.3448464, 0.6878246, -0.2204866, 0.2235123
4: -4.2197785, -3.3744671, -4.2200346, -3.3743250, -0.2723126, 0.2640680
5: 1.8260088, 2.1858191, 1.8226287, 2.1858647, -0.1202787, 0.1281731
6: -4.4170284, -3.4958227, -4.4170704, -3.4958315, -0.1768186, 0.1773901
7: -0.4345598, 0.0873711, -0.4351711, 0.0896150, -0.1332576, 0.1309779
8: -3.6474850, -1.5570610, -3.6532991, -1.5560799, -0.9024929, 0.9108275
9: -2.8998435, -1.1937230, -2.9008288, -1.1933756, -0.7733891, 0.7781873

Time for backsubstitution: 6.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 2139
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2605
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2426
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 361
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 2585
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 379
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 2170
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 3125
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2436
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 3010
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 2091
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 2507
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 2972
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 3538
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 2231
type: B, layer: 1, pos: 3127
type: B, layer: 1, pos: 2680
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3351
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2085
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 2187
type: B, layer: 1, pos: 2317
type: B, layer: 1, pos: 2070
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 2657
type: B, layer: 1, pos: 2451
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2550
type: B, layer: 1, pos: 2179
type: B, layer: 1, pos: 423
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 2213
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 2203
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 2331
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 2209
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 3525
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 2282
type: B, layer: 1, pos: 2408
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 2672
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 2400
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 2956
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 2435
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 2116
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 2595
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 3114
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 2671
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 2322
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 2997
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 2037
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 2250
type: B, layer: 1, pos: 3253
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2278
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 2926
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 3290
type: B, layer: 1, pos: 2081
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 2082
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 2925
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 2030
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 3514
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 2928
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 2927
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 438
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2084
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2244
type: B, layer: 1, pos: 2245
type: B, layer: 1, pos: 2246
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 2686
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 3149
type: B, layer: 1, pos: 3368

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 2377

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0929308, upper bound: 0.0929485
time: 102.05 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0929342, upper bound: 0.0930094
time: 104.16 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3.2714338, -1.3593209, -3.2709959, -1.3600047, -1.5909944, 1.5817692
1: -3.1564789, -1.3766823, -3.1554646, -1.3774745, -1.2098857, 1.1969340
2: -1.0849576, -0.6185258, -1.0851740, -0.6129243, -0.2452883, 0.2373389
3: 0.3421240, 0.6879890, 0.3422087, 0.6878478, -0.2239904, 0.2270348
4: -4.2205472, -3.3748817, -4.2206783, -3.3743186, -0.2728703, 0.2656101
5: 1.8220167, 2.1858909, 1.8193383, 2.1858692, -0.1238137, 0.1328155
6: -4.4171138, -3.4960284, -4.4171462, -3.4958208, -0.1767094, 0.1779521
7: -0.4381715, 0.0872500, -0.4381733, 0.0896272, -0.1373078, 0.1352489
8: -3.6489513, -1.5505898, -3.6533809, -1.5505326, -0.9175414, 0.9075365
9: -2.9022231, -1.1856561, -2.9008598, -1.1867669, -0.7922142, 0.7719066

Time for backsubstitution: 6.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 2139
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 2605
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2426
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 361
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 2585
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 379
type: B, layer: 1, pos: 2170
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 3125
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2436
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 3010
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 2091
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 2507
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 2972
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 3538
type: B, layer: 1, pos: 2231
type: B, layer: 1, pos: 3127
type: B, layer: 1, pos: 2680
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3351
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 2187
type: B, layer: 1, pos: 2085
type: B, layer: 1, pos: 2317
type: B, layer: 1, pos: 2070
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 2657
type: B, layer: 1, pos: 2451
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2550
type: B, layer: 1, pos: 2179
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 423
type: B, layer: 1, pos: 2213
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 2203
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 2331
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2209
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 3525
type: B, layer: 1, pos: 2408
type: B, layer: 1, pos: 2282
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 2672
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 2400
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 2956
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 2435
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 2581
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2116
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 2595
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 3114
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 2671
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2973
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 2322
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2997
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 2037
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 3253
type: B, layer: 1, pos: 2250
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2278
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 2926
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 3290
type: B, layer: 1, pos: 2081
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 2082
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 2925
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 2030
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 3514
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 2928
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 2927
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 438
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2084
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2244
type: B, layer: 1, pos: 2245
type: B, layer: 1, pos: 2246
type: B, layer: 1, pos: 2279
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 2686
type: B, layer: 1, pos: 2690
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 3149
type: B, layer: 1, pos: 3368

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2377

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930267, upper bound: 0.0929789
time: 14.45 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0930317, upper bound: 0.0930376
time: 35.21 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 56.35 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 56.35
Output dim: 5, lower bound: -0.0924985, upper bound: 0.0929428
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 56.35
Output dim: 5, lower bound: -0.0925046, upper bound: 0.0930022
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 56.35
Output dim: 5, lower bound: -0.0925963, upper bound: 0.0929735
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 56.35
Output dim: 5, lower bound: -0.0926005, upper bound: 0.0930345
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 56.35
Output dim: 5, lower bound: -0.0926587, upper bound: 0.0929439
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 56.35
Output dim: 5, lower bound: -0.0926624, upper bound: 0.0929458
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 56.35
Output dim: 5, lower bound: -0.0927561, upper bound: 0.0929774
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 56.35
Output dim: 5, lower bound: -0.0927603, upper bound: 0.0930356
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 56.35
Output dim: 5, lower bound: -0.0927708, upper bound: 0.0929465
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 56.35
Output dim: 5, lower bound: -0.0927747, upper bound: 0.0930109
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 56.35
Output dim: 5, lower bound: -0.0928674, upper bound: 0.0929789
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 56.35
Output dim: 5, lower bound: -0.0928713, upper bound: 0.0930336
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 56.35
Output dim: 5, lower bound: -0.0929308, upper bound: 0.0929485
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 56.35
Output dim: 5, lower bound: -0.0929342, upper bound: 0.0930094
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 56.35
Output dim: 5, lower bound: -0.0930267, upper bound: 0.0929789
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 56.35
Output dim: 5, lower bound: -0.0930317, upper bound: 0.0930376

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -3.2696252, -1.3752685, -3.2728353, -1.3741114, -1.5703731, 1.5742862
1: -3.1580329, -1.3908801, -3.1606627, -1.3897374, -1.1887329, 1.1939549
2: -1.0825151, -0.6212967, -1.0829173, -0.6209137, -0.2350170, 0.2352432
3: 0.3486159, 0.6883783, 0.3478733, 0.6886393, -0.2192057, 0.2196546
4: -4.2155452, -3.3810735, -4.2178764, -3.3801758, -0.2626640, 0.2635174
5: 1.8307624, 2.1839349, 1.8295695, 2.1853960, -0.1199602, 0.1194094
6: -4.4147654, -3.4962263, -4.4146657, -3.4960873, -0.1754786, 0.1749286
7: -0.4334029, 0.0871124, -0.4336857, 0.0869273, -0.1291166, 0.1293069
8: -3.6477242, -1.5586455, -3.6491261, -1.5593209, -0.8992602, 0.9051707
9: -2.9043789, -1.1950197, -2.9067161, -1.1939087, -0.7710726, 0.7771558

Time for backsubstitution: 6.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 2139
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 2426
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 394
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 379
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 2170
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 3125
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2436
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 3010
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 2507
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2972
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 3538
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 3127
type: A, layer: 1, pos: 2680
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3351
type: A, layer: 1, pos: 2085
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 2187
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 2070
type: A, layer: 1, pos: 2317
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 2657
type: A, layer: 1, pos: 2451
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 2550
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2179
type: A, layer: 1, pos: 423
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 2213
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 2676
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2331
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 2209
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 3525
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 2282
type: A, layer: 1, pos: 2408
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 2672
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 2400
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 2956
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 2435
type: A, layer: 1, pos: 2116
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 2595
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 3114
type: A, layer: 1, pos: 2671
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2322
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 2997
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2037
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 2250
type: A, layer: 1, pos: 3253
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 2278
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 2926
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 3290
type: A, layer: 1, pos: 2081
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 2082
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 2925
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 3514
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 2928
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 2927
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 438
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2084
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2244
type: A, layer: 1, pos: 2245
type: A, layer: 1, pos: 2246
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 2685
type: A, layer: 1, pos: 2686
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 3149
type: A, layer: 1, pos: 3368

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2387

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0923764, upper bound: 0.0928998
time: 17.80 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0924932, upper bound: 0.0929404
time: 12.90 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -3.2697234, -1.3731070, -3.2795794, -1.3715401, -1.5729797, 1.5840306
1: -3.1580765, -1.3888166, -3.1667621, -1.3872294, -1.1903651, 1.2022891
2: -1.0825512, -0.6211810, -1.0836244, -0.6207824, -0.2352595, 0.2366282
3: 0.3480959, 0.6883955, 0.3471907, 0.6892892, -0.2204033, 0.2201803
4: -4.2165751, -3.3810699, -4.2191620, -3.3779466, -0.2659836, 0.2638655
5: 1.8302736, 2.1839387, 1.8289324, 2.1859121, -0.1211641, 0.1198887
6: -4.4157143, -3.4962168, -4.4158192, -3.4938066, -0.1789837, 0.1751384
7: -0.4336753, 0.0872772, -0.4341780, 0.0871352, -0.1295789, 0.1299678
8: -3.6477540, -1.5577416, -3.6516716, -1.5581679, -0.8998402, 0.9085180
9: -2.9043903, -1.1942639, -2.9085717, -1.1929464, -0.7713066, 0.7804191

Time for backsubstitution: 6.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 2139
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 2426
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 394
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 379
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 2170
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 3125
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2436
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 3010
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 2507
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2972
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 3538
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 3127
type: A, layer: 1, pos: 2680
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3351
type: A, layer: 1, pos: 2085
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 2187
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 2070
type: A, layer: 1, pos: 2317
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 2657
type: A, layer: 1, pos: 2451
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 2550
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2179
type: A, layer: 1, pos: 423
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 2213
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 2676
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2331
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 2209
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 3525
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 2282
type: A, layer: 1, pos: 2408
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 2672
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 2400
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 2956
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 2435
type: A, layer: 1, pos: 2116
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 2595
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 3114
type: A, layer: 1, pos: 2671
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2322
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 2997
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2037
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 2250
type: A, layer: 1, pos: 3253
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 2278
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 2926
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 3290
type: A, layer: 1, pos: 2081
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 2082
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 2925
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 3514
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 2928
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 2927
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 438
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2084
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2244
type: A, layer: 1, pos: 2245
type: A, layer: 1, pos: 2246
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 2685
type: A, layer: 1, pos: 2686
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 3149
type: A, layer: 1, pos: 3368

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2387

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0923855, upper bound: 0.0929662
time: 8.01 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0924983, upper bound: 0.0929991
time: 8.82 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -3.2712257, -1.3675094, -3.2730141, -1.3675628, -1.5814199, 1.5787964
1: -3.1604698, -1.3813653, -3.1607792, -1.3819900, -1.2060261, 1.1938217
2: -1.0826359, -0.6209013, -1.0830235, -0.6206028, -0.2359155, 0.2354132
3: 0.3454635, 0.6880461, 0.3452379, 0.6886624, -0.2227049, 0.2231740
4: -4.2163148, -3.3814881, -4.2185197, -3.3801687, -0.2632258, 0.2650550
5: 1.8267734, 2.1840065, 1.8262715, 2.1854005, -0.1234944, 0.1240858
6: -4.4148512, -3.4964304, -4.4147420, -3.4960771, -0.1753693, 0.1754903
7: -0.4370141, 0.0869914, -0.4366857, 0.0869394, -0.1331654, 0.1335723
8: -3.6491909, -1.5521741, -3.6492095, -1.5537741, -0.9143090, 0.9018812
9: -2.9067581, -1.1869531, -2.9067478, -1.1873002, -0.7898964, 0.7708757

Time for backsubstitution: 6.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 2139
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 2426
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 394
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 379
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 2170
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 3125
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2436
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 3010
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 2507
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2972
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 3538
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 3127
type: A, layer: 1, pos: 2680
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3351
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 2187
type: A, layer: 1, pos: 2085
type: A, layer: 1, pos: 2317
type: A, layer: 1, pos: 2070
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 2657
type: A, layer: 1, pos: 2451
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2550
type: A, layer: 1, pos: 2179
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 423
type: A, layer: 1, pos: 2213
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 2676
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 2331
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 2209
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 3525
type: A, layer: 1, pos: 2408
type: A, layer: 1, pos: 2282
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 2672
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 2400
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 2956
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 2435
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 2116
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 2595
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 3114
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2671
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 2322
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 2997
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2037
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 3253
type: A, layer: 1, pos: 2250
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 2278
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 2926
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 3290
type: A, layer: 1, pos: 2081
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 2082
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 2925
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 3514
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 2928
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 2927
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 438
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2084
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2244
type: A, layer: 1, pos: 2245
type: A, layer: 1, pos: 2246
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 2685
type: A, layer: 1, pos: 2686
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 3149
type: A, layer: 1, pos: 3368

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 2387

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0924849, upper bound: 0.0929328
time: 139.61 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0925885, upper bound: 0.0929652
time: 11.27 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -3.2713237, -1.3653479, -3.2797577, -1.3649902, -1.5840261, 1.5885396
1: -3.1605127, -1.3793006, -3.1668787, -1.3794813, -1.2076588, 1.2021558
2: -1.0826716, -0.6207857, -1.0837302, -0.6204715, -0.2361578, 0.2367979
3: 0.3449416, 0.6880631, 0.3445527, 0.6893123, -0.2239037, 0.2237016
4: -4.2173443, -3.3814847, -4.2198067, -3.3779397, -0.2665445, 0.2654016
5: 1.8262833, 2.1840103, 1.8256325, 2.1859164, -0.1246984, 0.1245651
6: -4.4158001, -3.4964230, -4.4158955, -3.4937959, -0.1788744, 0.1757002
7: -0.4372868, 0.0871564, -0.4371686, 0.0871473, -0.1336265, 0.1342288
8: -3.6492207, -1.5512702, -3.6517534, -1.5526206, -0.9148885, 0.9052287
9: -2.9067695, -1.1861973, -2.9086032, -1.1863389, -0.7901309, 0.7741381

Time for backsubstitution: 6.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 2139
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 2426
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 394
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 379
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 2170
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 3125
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2436
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 3010
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 2507
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2972
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 3538
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 3127
type: A, layer: 1, pos: 2680
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3351
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 2187
type: A, layer: 1, pos: 2085
type: A, layer: 1, pos: 2317
type: A, layer: 1, pos: 2070
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 2657
type: A, layer: 1, pos: 2451
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2550
type: A, layer: 1, pos: 2179
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 423
type: A, layer: 1, pos: 2213
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 2676
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 2331
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 2209
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 3525
type: A, layer: 1, pos: 2408
type: A, layer: 1, pos: 2282
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 2672
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 2400
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 2956
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 2435
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 2116
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 2595
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 3114
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2671
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 2322
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 2997
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2037
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 3253
type: A, layer: 1, pos: 2250
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 2278
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 2926
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 3290
type: A, layer: 1, pos: 2081
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 2082
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 2925
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 3514
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 2928
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 2927
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 438
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2084
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2244
type: A, layer: 1, pos: 2245
type: A, layer: 1, pos: 2246
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 2685
type: A, layer: 1, pos: 2686
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 3149
type: A, layer: 1, pos: 3368

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2387

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0924898, upper bound: 0.0929962
time: 8.39 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0925950, upper bound: 0.0930220
time: 145.74 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -3.2673352, -1.3752670, -3.2701185, -1.3748207, -1.5686750, 1.5739150
1: -3.1536112, -1.3908787, -3.1552336, -1.3905556, -1.1868743, 1.1938000
2: -1.0835612, -0.6212898, -1.0845730, -0.6153240, -0.2413677, 0.2358490
3: 0.3486171, 0.6876047, 0.3480247, 0.6877219, -0.2190639, 0.2200582
4: -4.2157373, -3.3810596, -4.2182131, -3.3795974, -0.2633404, 0.2637497
5: 1.8307631, 2.1842697, 1.8268899, 2.1858420, -0.1201426, 0.1223425
6: -4.4147949, -3.4962263, -4.4147081, -3.4958825, -0.1755573, 0.1749540
7: -0.4338262, 0.0871141, -0.4342700, 0.0893070, -0.1319185, 0.1294420
8: -3.6477361, -1.5578237, -3.6534648, -1.5581119, -0.8996433, 0.9096671
9: -2.8998568, -1.1950145, -2.9008284, -1.1950381, -0.7704574, 0.7770567

Time for backsubstitution: 6.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 2139
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 2426
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 394
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 379
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 2170
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 3125
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2436
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 3010
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 2507
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2972
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 3538
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 3127
type: A, layer: 1, pos: 2680
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3351
type: A, layer: 1, pos: 2085
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 2187
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 2070
type: A, layer: 1, pos: 2317
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 2657
type: A, layer: 1, pos: 2451
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 2550
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2179
type: A, layer: 1, pos: 423
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 2213
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 2676
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2331
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 2209
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 3525
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 2282
type: A, layer: 1, pos: 2408
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 2672
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 2400
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 2956
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 2435
type: A, layer: 1, pos: 2116
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 2595
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 3114
type: A, layer: 1, pos: 2671
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2322
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 2997
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2037
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 2250
type: A, layer: 1, pos: 3253
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 2278
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 2926
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 3290
type: A, layer: 1, pos: 2081
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 2082
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 2925
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 3514
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 2928
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 2927
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 438
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2084
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2244
type: A, layer: 1, pos: 2245
type: A, layer: 1, pos: 2246
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 2685
type: A, layer: 1, pos: 2686
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 3149
type: A, layer: 1, pos: 3368

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2387

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0925396, upper bound: 0.0929019
time: 101.59 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0926527, upper bound: 0.0929406
time: 108.77 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -3.2674329, -1.3731062, -3.2768652, -1.3722484, -1.5712850, 1.5836585
1: -3.1536555, -1.3888144, -3.1613350, -1.3880479, -1.1885083, 1.2021341
2: -1.0835977, -0.6211742, -1.0852799, -0.6151927, -0.2416103, 0.2372338
3: 0.3480970, 0.6876217, 0.3473425, 0.6883718, -0.2202614, 0.2205835
4: -4.2167683, -3.3810565, -4.2194991, -3.3773689, -0.2666600, 0.2640980
5: 1.8302745, 2.1842737, 1.8262533, 2.1863585, -0.1213464, 0.1228199
6: -4.4157438, -3.4962187, -4.4158602, -3.4936011, -0.1790624, 0.1751638
7: -0.4340985, 0.0872791, -0.4347629, 0.0895149, -0.1323808, 0.1301031
8: -3.6477671, -1.5569198, -3.6560094, -1.5569584, -0.9002231, 0.9130144
9: -2.8998687, -1.1942587, -2.9026847, -1.1940753, -0.7706916, 0.7803198

Time for backsubstitution: 6.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 2139
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 2426
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 394
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 379
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 2170
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 3125
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2436
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 3010
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 2507
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2972
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 3538
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 3127
type: A, layer: 1, pos: 2680
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3351
type: A, layer: 1, pos: 2085
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 2187
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 2070
type: A, layer: 1, pos: 2317
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 2657
type: A, layer: 1, pos: 2451
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 2550
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2179
type: A, layer: 1, pos: 423
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 2213
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 2676
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2331
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 2209
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 3525
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 2282
type: A, layer: 1, pos: 2408
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 2672
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 2400
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 2956
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 2435
type: A, layer: 1, pos: 2116
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 2595
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 3114
type: A, layer: 1, pos: 2671
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2322
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 2997
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2037
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 2250
type: A, layer: 1, pos: 3253
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 2278
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 2926
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 3290
type: A, layer: 1, pos: 2081
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 2082
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 2925
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 3514
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 2928
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 2927
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 438
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2084
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2244
type: A, layer: 1, pos: 2245
type: A, layer: 1, pos: 2246
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 2685
type: A, layer: 1, pos: 2686
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 3149
type: A, layer: 1, pos: 3368

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2387

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0925476, upper bound: 0.0929615
time: 330.31 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0926556, upper bound: 0.0930038
time: 6.69 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -3.2689357, -1.3675082, -3.2703004, -1.3682714, -1.5797288, 1.5784245
1: -3.1560500, -1.3813641, -3.1553526, -1.3828092, -1.2041717, 1.1936660
2: -1.0836829, -0.6208944, -1.0846798, -0.6150131, -0.2422669, 0.2360186
3: 0.3454646, 0.6872723, 0.3453971, 0.6877450, -0.2225631, 0.2235662
4: -4.2165070, -3.3814743, -4.2188573, -3.3795905, -0.2639023, 0.2652875
5: 1.8267741, 2.1843412, 1.8236029, 2.1858468, -0.1236766, 0.1269843
6: -4.4148803, -3.4964316, -4.4147844, -3.4958720, -0.1754482, 0.1755156
7: -0.4374372, 0.0869933, -0.4372714, 0.0893193, -0.1359673, 0.1337075
8: -3.6492028, -1.5513532, -3.6535470, -1.5525646, -0.9146919, 0.9063772
9: -2.9022365, -1.1869471, -2.9008598, -1.1884291, -0.7892824, 0.7707765

Time for backsubstitution: 6.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 2139
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 2426
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 394
type: A, layer: 1, pos: 379
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 2170
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 3125
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2436
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 3010
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 2507
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2972
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 3538
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 3127
type: A, layer: 1, pos: 2680
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3351
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 2187
type: A, layer: 1, pos: 2085
type: A, layer: 1, pos: 2317
type: A, layer: 1, pos: 2070
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 2657
type: A, layer: 1, pos: 2451
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2550
type: A, layer: 1, pos: 2179
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 423
type: A, layer: 1, pos: 2213
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 2676
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 2331
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 2209
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 3525
type: A, layer: 1, pos: 2408
type: A, layer: 1, pos: 2282
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 2672
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 2400
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 2956
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 2435
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 2116
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 2595
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 3114
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2671
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 2322
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 2997
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2037
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 3253
type: A, layer: 1, pos: 2250
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 2278
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 2926
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 3290
type: A, layer: 1, pos: 2081
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 2082
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 2925
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 3514
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 2928
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 2927
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 438
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2084
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2244
type: A, layer: 1, pos: 2245
type: A, layer: 1, pos: 2246
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 2685
type: A, layer: 1, pos: 2686
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 3149
type: A, layer: 1, pos: 3368

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2387

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0926460, upper bound: 0.0929404
time: 6.50 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0927487, upper bound: 0.0929691
time: 13.67 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -3.2690341, -1.3653462, -3.2770448, -1.3656995, -1.5823381, 1.5881691
1: -3.1560931, -1.3792999, -3.1614532, -1.3802998, -1.2058059, 1.2020011
2: -1.0837188, -0.6207789, -1.0853866, -0.6148817, -0.2425093, 0.2374032
3: 0.3449428, 0.6872895, 0.3447125, 0.6883948, -0.2237618, 0.2240931
4: -4.2175364, -3.3814709, -4.2201428, -3.3773623, -0.2672209, 0.2656341
5: 1.8262842, 2.1843450, 1.8229644, 2.1863627, -0.1248807, 0.1274613
6: -4.4158297, -3.4964228, -4.4159369, -3.4935915, -0.1789533, 0.1757256
7: -0.4377095, 0.0871582, -0.4377545, 0.0895271, -0.1364283, 0.1343640
8: -3.6492331, -1.5504496, -3.6560912, -1.5514109, -0.9152718, 0.9097246
9: -2.9022477, -1.1861920, -2.9027157, -1.1874673, -0.7895167, 0.7740387

Time for backsubstitution: 6.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 2139
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 2426
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 394
type: A, layer: 1, pos: 379
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 2170
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 3125
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2436
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 3010
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 2507
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2972
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 3538
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 3127
type: A, layer: 1, pos: 2680
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3351
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 2187
type: A, layer: 1, pos: 2085
type: A, layer: 1, pos: 2317
type: A, layer: 1, pos: 2070
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 2657
type: A, layer: 1, pos: 2451
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2550
type: A, layer: 1, pos: 2179
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 423
type: A, layer: 1, pos: 2213
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 2676
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 2331
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 2209
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 3525
type: A, layer: 1, pos: 2408
type: A, layer: 1, pos: 2282
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 2672
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 2400
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 2956
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 2435
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 2116
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 2595
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 3114
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2671
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 2322
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 2997
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2037
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 3253
type: A, layer: 1, pos: 2250
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 2278
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 2926
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 3290
type: A, layer: 1, pos: 2081
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 2082
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 2925
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 3514
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 2928
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 2927
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 438
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2084
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2244
type: A, layer: 1, pos: 2245
type: A, layer: 1, pos: 2246
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 2685
type: A, layer: 1, pos: 2686
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 3149
type: A, layer: 1, pos: 3368

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2387

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0926505, upper bound: 0.0929357
time: 167.94 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0927537, upper bound: 0.0930279
time: 35.95 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -3.2719333, -1.3713582, -3.2733002, -1.3710010, -1.5759413, 1.5728405
1: -3.1583750, -1.3900654, -3.1606688, -1.3890910, -1.1898686, 1.1933947
2: -1.0837131, -0.6194059, -1.0833164, -0.6193974, -0.2372909, 0.2359214
3: 0.3464974, 0.6890488, 0.3461632, 0.6886850, -0.2193266, 0.2215418
4: -4.2180996, -3.3744860, -4.2178926, -3.3749087, -0.2700483, 0.2620077
5: 1.8271544, 2.1854773, 1.8266984, 2.1854098, -0.1188371, 0.1237235
6: -4.4155760, -3.4958546, -4.4153128, -3.4960761, -0.1753373, 0.1756743
7: -0.4334015, 0.0871212, -0.4336884, 0.0869343, -0.1291211, 0.1293246
8: -3.6474133, -1.5594339, -3.6488907, -1.5591908, -0.9002523, 0.9047800
9: -2.9043429, -1.1950510, -2.9066896, -1.1938689, -0.7723328, 0.7768425

Time for backsubstitution: 6.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 2139
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 2426
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 394
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 379
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 2170
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 3125
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2436
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 3010
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 2507
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2972
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 3538
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 3127
type: A, layer: 1, pos: 2680
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3351
type: A, layer: 1, pos: 2085
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 2187
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 2070
type: A, layer: 1, pos: 2317
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 2657
type: A, layer: 1, pos: 2451
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 2550
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2179
type: A, layer: 1, pos: 423
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 2213
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 2676
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2331
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 2209
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 3525
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 2282
type: A, layer: 1, pos: 2408
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 2672
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 2400
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 2956
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 2435
type: A, layer: 1, pos: 2116
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 2595
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 3114
type: A, layer: 1, pos: 2671
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2322
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 2997
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2037
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 2250
type: A, layer: 1, pos: 3253
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 2278
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 2926
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 3290
type: A, layer: 1, pos: 2081
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 2082
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 2925
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 3514
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 2928
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 2927
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 438
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2084
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2244
type: A, layer: 1, pos: 2245
type: A, layer: 1, pos: 2246
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 2685
type: A, layer: 1, pos: 2686
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 3149
type: A, layer: 1, pos: 3368

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2387

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0926473, upper bound: 0.0929070
time: 25.32 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0927632, upper bound: 0.0929420
time: 116.08 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -3.2720330, -1.3691964, -3.2800443, -1.3684282, -1.5785477, 1.5825846
1: -3.1584194, -1.3880017, -3.1667690, -1.3865824, -1.1915011, 1.2017281
2: -1.0837494, -0.6192902, -1.0840230, -0.6192661, -0.2375336, 0.2373068
3: 0.3459768, 0.6890659, 0.3454804, 0.6893349, -0.2205243, 0.2220682
4: -4.2191296, -3.3744826, -4.2191777, -3.3726795, -0.2733677, 0.2623560
5: 1.8266658, 2.1854808, 1.8260614, 2.1859260, -0.1200411, 0.1242028
6: -4.4165249, -3.4958463, -4.4164643, -3.4937959, -0.1788424, 0.1758842
7: -0.4336739, 0.0872862, -0.4341809, 0.0871421, -0.1295834, 0.1299855
8: -3.6474442, -1.5585303, -3.6514351, -1.5580378, -0.9008317, 0.9081274
9: -2.9043550, -1.1942949, -2.9085453, -1.1929080, -0.7725668, 0.7801059

Time for backsubstitution: 6.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 2139
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 2426
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 394
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 379
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 2170
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 3125
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2436
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 3010
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 2507
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 2972
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 3538
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 3127
type: A, layer: 1, pos: 2680
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 3351
type: A, layer: 1, pos: 2085
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 2187
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 2070
type: A, layer: 1, pos: 2317
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 2657
type: A, layer: 1, pos: 2451
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 2550
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2179
type: A, layer: 1, pos: 423
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 2213
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 2676
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2331
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 2209
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 3525
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 2282
type: A, layer: 1, pos: 2408
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 2672
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 2400
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 2956
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 2581
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 2435
type: A, layer: 1, pos: 2116
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 2595
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 3114
type: A, layer: 1, pos: 2671
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2973
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 2322
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 2997
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2037
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 2250
type: A, layer: 1, pos: 3253
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 2278
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 2926
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 3290
type: A, layer: 1, pos: 2081
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 2082
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 2925
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 3514
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 2928
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 2927
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 438
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2084
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2244
type: A, layer: 1, pos: 2245
type: A, layer: 1, pos: 2246
type: A, layer: 1, pos: 2279
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 2685
type: A, layer: 1, pos: 2686
type: A, layer: 1, pos: 2690
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 3149
type: A, layer: 1, pos: 3368

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2387

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0926555, upper bound: 0.0929689
time: 8.87 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0927661, upper bound: 0.0930036
time: 14.72 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -3.2735360, -1.3635995, -3.2734807, -1.3644514, -1.5869901, 1.5773504
1: -3.1608133, -1.3805501, -3.1607871, -1.3813441, -1.2071620, 1.1932602
2: -1.0838342, -0.6190104, -1.0834229, -0.6190864, -0.2381897, 0.2360913
3: 0.3433382, 0.6887165, 0.3435220, 0.6887081, -0.2228252, 0.2250697
4: -4.2188687, -3.3749003, -4.2185364, -3.3749022, -0.2706062, 0.2635486
5: 1.8231661, 2.1855488, 1.8234010, 2.1854146, -0.1223715, 0.1283997
6: -4.4156609, -3.4960599, -4.4153886, -3.4960661, -0.1752279, 0.1762363
7: -0.4370123, 0.0870001, -0.4366884, 0.0869465, -0.1331697, 0.1335900
8: -3.6488805, -1.5529630, -3.6489735, -1.5536432, -0.9153007, 0.9014903
9: -2.9067223, -1.1869841, -2.9067209, -1.1872606, -0.7911564, 0.7705622

Time for backsubstitution: 6.55 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 85.21 + 3517.69 = 3602.89 seconds
