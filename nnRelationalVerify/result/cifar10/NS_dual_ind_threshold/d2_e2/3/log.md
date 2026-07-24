## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 3)
Time budget: 3600 seconds
Split limit: 100
Threshold: 0.029325626400000002


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3.0459356, -1.4632723, -3.0459356, -1.4632723, -0.5215368, 0.5215368)
1: (-2.2329955, -0.5492516, -2.2329955, -0.5492516, -0.5965176, 0.5965177)
2: (-2.5440857, -1.8995905, -2.5440857, -1.8995905, -0.1443744, 0.1443744)
3: (-0.8581853, -0.5421180, -0.8581853, -0.5421180, -0.0767322, 0.0767322)
4: (-2.4013999, -1.4624283, -2.4013999, -1.4624283, -0.2031510, 0.2031510)
5: (-0.4060171, -0.1588888, -0.4060171, -0.1588888, -0.0569329, 0.0569329)
6: (-1.2678086, -0.6738276, -1.2678086, -0.6738276, -0.1527211, 0.1527211)
7: (-0.5173369, 0.2315183, -0.5173369, 0.2315183, -0.1320915, 0.1320915)
8: (-3.5481367, -1.8033185, -3.5481367, -1.8033185, -0.6374951, 0.6374950)
9: (-4.2394500, -2.5951118, -4.2394500, -2.5951118, -0.5994052, 0.5994052)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.92 + 37.26 = 45.18 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0294424, upper bound: 0.0294492

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 328
type: A, layer: 1, pos: 3491
type: A, layer: 1, pos: 318
type: A, layer: 1, pos: 2426
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 3477
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 2635
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 3498
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 2125
type: A, layer: 1, pos: 3545
type: A, layer: 1, pos: 2391
type: A, layer: 1, pos: 2211
type: A, layer: 1, pos: 2365
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 3041
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 2210
type: A, layer: 1, pos: 2359
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 2199
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2108
type: A, layer: 1, pos: 304
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 3531
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 431
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 313
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 110
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 3220
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 2404
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 2976
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2187
type: A, layer: 1, pos: 2286
type: A, layer: 1, pos: 3285
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 2086
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 3267
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 2293
type: A, layer: 1, pos: 2292
type: A, layer: 1, pos: 3296
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 2492
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 2491
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 3238
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2506
type: A, layer: 1, pos: 2041
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2073
type: A, layer: 1, pos: 2610
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 2625
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 2085
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2204
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2687
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 328

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0292298, upper bound: 0.0294525
time: 6.66 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0294421, upper bound: 0.0294476
time: 216.24 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 222.99 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 222.99
Output dim: 3, lower bound: -0.0292298, upper bound: 0.0294525
NS_A2, status: Status.UNKNOWN, split count: 1, time: 222.99
Output dim: 3, lower bound: -0.0294421, upper bound: 0.0294476

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -3.0449421, -1.4668889, -3.0457730, -1.4659410, -0.5177312, 0.5176058
1: -2.2310481, -0.5580006, -2.2329919, -0.5556936, -0.5881437, 0.5877935
2: -2.5385396, -1.9009354, -2.5400281, -1.8996810, -0.1387711, 0.1389538
3: -0.8551480, -0.5428804, -0.8560658, -0.5421966, -0.0735998, 0.0738094
4: -2.4012167, -1.4626590, -2.4012592, -1.4625981, -0.2026515, 0.2027002
5: -0.4031162, -0.1595428, -0.4040057, -0.1589177, -0.0539828, 0.0542367
6: -1.2653521, -0.6743174, -1.2661791, -0.6738329, -0.1502795, 0.1505854
7: -0.5171037, 0.2311541, -0.5168120, 0.2312114, -0.1309425, 0.1307484
8: -3.5468371, -1.8089483, -3.5481000, -1.8074644, -0.6320418, 0.6318336
9: -4.2384157, -2.5998728, -4.2394500, -2.5986001, -0.5949151, 0.5947379

Time for backsubstitution: 6.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3491
type: B, layer: 1, pos: 2426
type: B, layer: 1, pos: 318
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 3477
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 2635
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 3498
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2541
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 3545
type: B, layer: 1, pos: 2391
type: B, layer: 1, pos: 2211
type: B, layer: 1, pos: 2365
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 3041
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 2539
type: B, layer: 1, pos: 2210
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 2231
type: B, layer: 1, pos: 2199
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 328
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2108
type: B, layer: 1, pos: 304
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2117
type: B, layer: 1, pos: 2203
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 3531
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 431
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 313
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 110
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 3220
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 2404
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 2557
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 3462
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 2087
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2976
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2187
type: B, layer: 1, pos: 2286
type: B, layer: 1, pos: 3285
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 2086
type: B, layer: 1, pos: 2600
type: B, layer: 1, pos: 3267
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 2293
type: B, layer: 1, pos: 2292
type: B, layer: 1, pos: 3296
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2052
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 2492
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 2491
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 2190
type: B, layer: 1, pos: 3238
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2506
type: B, layer: 1, pos: 2041
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2073
type: B, layer: 1, pos: 2610
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2056
type: B, layer: 1, pos: 2051
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 2625
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 2085
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2069
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2204
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2234
type: B, layer: 1, pos: 2243
type: B, layer: 1, pos: 2687
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 3149

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3491

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0291059, upper bound: 0.0294399
time: 35.17 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0292273, upper bound: 0.0294402
time: 285.79 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -3.0459354, -1.4632781, -3.0459354, -1.4632759, -0.5215234, 0.5174628
1: -2.2329955, -0.5492582, -2.2329955, -0.5492568, -0.5965097, 0.5863916
2: -2.5440860, -1.8995907, -2.5440860, -1.8995905, -0.1380984, 0.1443650
3: -0.8581852, -0.5421181, -0.8581852, -0.5421180, -0.0733798, 0.0767317
4: -2.4014001, -1.4624451, -2.4013999, -1.4624398, -0.2031495, 0.2027971
5: -0.4060169, -0.1588888, -0.4060169, -0.1588888, -0.0537507, 0.0569316
6: -1.2678071, -0.6738276, -1.2678075, -0.6738276, -0.1502436, 0.1527193
7: -0.5173369, 0.2314389, -0.5173370, 0.2314639, -0.1320737, 0.1316717
8: -3.5481367, -1.8033228, -3.5481365, -1.8033218, -0.6374876, 0.6309783
9: -4.2394500, -2.5951154, -4.2394500, -2.5951145, -0.5993957, 0.5940605

Time for backsubstitution: 6.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3491
type: B, layer: 1, pos: 318
type: B, layer: 1, pos: 2426
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 3477
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 2635
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 3498
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2541
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 3545
type: B, layer: 1, pos: 2391
type: B, layer: 1, pos: 2211
type: B, layer: 1, pos: 2365
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 3041
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 2539
type: B, layer: 1, pos: 2210
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 2231
type: B, layer: 1, pos: 2199
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 328
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2108
type: B, layer: 1, pos: 304
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2117
type: B, layer: 1, pos: 2203
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 3531
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 431
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 313
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 110
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 3220
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 2404
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 2557
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 3462
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 2087
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2976
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2187
type: B, layer: 1, pos: 2286
type: B, layer: 1, pos: 3285
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 2086
type: B, layer: 1, pos: 2600
type: B, layer: 1, pos: 3267
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 2293
type: B, layer: 1, pos: 2292
type: B, layer: 1, pos: 3296
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2052
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 2492
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 2491
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 2190
type: B, layer: 1, pos: 3238
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2506
type: B, layer: 1, pos: 2041
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2073
type: B, layer: 1, pos: 2610
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2056
type: B, layer: 1, pos: 2051
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 2625
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 2085
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2069
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2204
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2234
type: B, layer: 1, pos: 2243
type: B, layer: 1, pos: 2687
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 3149

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3491

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0293186, upper bound: 0.0294553
time: 7.63 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0294407, upper bound: 0.0294524
time: 13.68 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 27.79 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 27.79
Output dim: 3, lower bound: -0.0291059, upper bound: 0.0294399
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 27.79
Output dim: 3, lower bound: -0.0292273, upper bound: 0.0294402
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 27.79
Output dim: 3, lower bound: -0.0293186, upper bound: 0.0294553
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 27.79
Output dim: 3, lower bound: -0.0294407, upper bound: 0.0294524

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -3.0443416, -1.4668889, -3.0450244, -1.4659410, -0.5170249, 0.5167260
1: -2.2297013, -0.5580006, -2.2313156, -0.5556936, -0.5866838, 0.5859754
2: -2.5385153, -1.9027719, -2.5399978, -1.9019639, -0.1361945, 0.1368691
3: -0.8551478, -0.5436978, -0.8560658, -0.5432127, -0.0726064, 0.0729931
4: -2.4008656, -1.4627416, -2.4008455, -1.4627025, -0.2019720, 0.2020116
5: -0.4031162, -0.1602022, -0.4040057, -0.1597382, -0.0533065, 0.0536929
6: -1.2653519, -0.6753010, -1.2661785, -0.6750348, -0.1489868, 0.1495479
7: -0.5170966, 0.2307727, -0.5168029, 0.2307588, -0.1303081, 0.1302363
8: -3.5462170, -1.8089483, -3.5473280, -1.8074644, -0.6313608, 0.6309854
9: -4.2379541, -2.6000111, -4.2388754, -2.5987732, -0.5940005, 0.5937990

Time for backsubstitution: 6.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 318
type: A, layer: 1, pos: 2426
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 3477
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 2635
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 3498
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 2125
type: A, layer: 1, pos: 3545
type: A, layer: 1, pos: 2391
type: A, layer: 1, pos: 2211
type: A, layer: 1, pos: 2365
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 3041
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 2210
type: A, layer: 1, pos: 2359
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 2199
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2108
type: A, layer: 1, pos: 304
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 3531
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 3491
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 431
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 313
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 110
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 3220
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 2404
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 2976
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2187
type: A, layer: 1, pos: 3285
type: A, layer: 1, pos: 2286
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2086
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 3267
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 2293
type: A, layer: 1, pos: 2292
type: A, layer: 1, pos: 3296
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 2492
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 2491
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 3238
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2041
type: A, layer: 1, pos: 2506
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 2073
type: A, layer: 1, pos: 2610
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 2625
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 2085
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2204
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2687
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 318

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0289528, upper bound: 0.0294519
time: 5.93 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0291052, upper bound: 0.0294372
time: 36.90 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -3.0446737, -1.4668889, -3.0457819, -1.4654415, -0.5179657, 0.5175412
1: -2.2303057, -0.5580006, -2.2323241, -0.5533524, -0.5892054, 0.5866023
2: -2.5385396, -1.9009377, -2.5421045, -1.8993973, -0.1377176, 0.1409609
3: -0.8551480, -0.5428846, -0.8569499, -0.5420615, -0.0732169, 0.0745358
4: -2.4011366, -1.4626591, -2.4012461, -1.4620888, -0.2028628, 0.2024525
5: -0.4031162, -0.1595471, -0.4048459, -0.1588775, -0.0536479, 0.0548926
6: -1.2653520, -0.6745050, -1.2668812, -0.6738235, -0.1498486, 0.1507781
7: -0.5171037, 0.2310746, -0.5178394, 0.2312266, -0.1306122, 0.1319516
8: -3.5465145, -1.8089483, -3.5478086, -1.8067560, -0.6324319, 0.6313479
9: -4.2381358, -2.5998733, -4.2391052, -2.5978429, -0.5950168, 0.5942260

Time for backsubstitution: 6.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 318
type: A, layer: 1, pos: 2426
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 3477
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 2635
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 3498
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 2125
type: A, layer: 1, pos: 3545
type: A, layer: 1, pos: 2391
type: A, layer: 1, pos: 2211
type: A, layer: 1, pos: 2365
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 3041
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 2210
type: A, layer: 1, pos: 2359
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 2199
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2108
type: A, layer: 1, pos: 304
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 3531
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 431
type: A, layer: 1, pos: 3491
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 313
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 110
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 3220
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 2404
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 2976
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2187
type: A, layer: 1, pos: 3285
type: A, layer: 1, pos: 2286
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2086
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 3267
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 2293
type: A, layer: 1, pos: 2292
type: A, layer: 1, pos: 3296
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 2492
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 2491
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 3238
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2041
type: A, layer: 1, pos: 2506
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 2073
type: A, layer: 1, pos: 2610
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 2625
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 2085
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2204
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2687
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 318

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0290706, upper bound: 0.0294425
time: 58.07 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0292262, upper bound: 0.0294428
time: 95.67 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -3.0453329, -1.4632781, -3.0451865, -1.4632759, -0.5208174, 0.5165837
1: -2.2316480, -0.5492582, -2.2313194, -0.5492568, -0.5950496, 0.5845734
2: -2.5440617, -1.9014269, -2.5440557, -1.9018734, -0.1355219, 0.1422805
3: -0.8581852, -0.5429351, -0.8581853, -0.5431339, -0.0723867, 0.0759159
4: -2.4010489, -1.4625280, -2.4009867, -1.4625444, -0.2024709, 0.2021097
5: -0.4060169, -0.1595483, -0.4060169, -0.1597094, -0.0530745, 0.0563878
6: -1.2678066, -0.6748110, -1.2678071, -0.6750294, -0.1489508, 0.1516817
7: -0.5173295, 0.2310576, -0.5173278, 0.2310115, -0.1314386, 0.1311599
8: -3.5475171, -1.8033228, -3.5473657, -1.8033218, -0.6368067, 0.6301303
9: -4.2389889, -2.5952537, -4.2388754, -2.5952880, -0.5984811, 0.5931219

Time for backsubstitution: 6.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 318
type: A, layer: 1, pos: 2426
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 3477
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 2635
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 3498
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 2125
type: A, layer: 1, pos: 3545
type: A, layer: 1, pos: 2391
type: A, layer: 1, pos: 2365
type: A, layer: 1, pos: 2211
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 3041
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 2210
type: A, layer: 1, pos: 2359
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 2199
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2108
type: A, layer: 1, pos: 304
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 3531
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 3491
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 431
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 313
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 110
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 3220
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 2404
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 2976
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2187
type: A, layer: 1, pos: 2286
type: A, layer: 1, pos: 3285
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 2086
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 3267
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 2293
type: A, layer: 1, pos: 2292
type: A, layer: 1, pos: 3296
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 2492
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 2491
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 3238
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2506
type: A, layer: 1, pos: 2041
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2073
type: A, layer: 1, pos: 2610
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 2625
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 2085
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2204
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2687
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 318

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0291616, upper bound: 0.0294519
time: 10.55 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0293184, upper bound: 0.0294546
time: 9.18 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -3.0456669, -1.4632781, -3.0459425, -1.4627764, -0.5217585, 0.5174009
1: -2.2322528, -0.5492582, -2.2323275, -0.5469141, -0.5975714, 0.5852003
2: -2.5440855, -1.8995929, -2.5461631, -1.8993068, -0.1370449, 0.1463732
3: -0.8581852, -0.5421221, -0.8590692, -0.5419831, -0.0729969, 0.0774582
4: -2.4013200, -1.4624453, -2.4013863, -1.4619305, -0.2033608, 0.2025523
5: -0.4060169, -0.1588931, -0.4068570, -0.1588487, -0.0534161, 0.0575875
6: -1.2678069, -0.6740150, -1.2685101, -0.6738180, -0.1498126, 0.1529120
7: -0.5173368, 0.2313595, -0.5183629, 0.2314792, -0.1317434, 0.1328831
8: -3.5478139, -1.8033228, -3.5478463, -1.8026135, -0.6378779, 0.6304929
9: -4.2391701, -2.5951169, -4.2391052, -2.5943575, -0.5994977, 0.5935487

Time for backsubstitution: 6.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 318
type: A, layer: 1, pos: 2426
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 3477
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 2635
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 3498
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 2125
type: A, layer: 1, pos: 3545
type: A, layer: 1, pos: 2391
type: A, layer: 1, pos: 2211
type: A, layer: 1, pos: 2365
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 3041
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 2210
type: A, layer: 1, pos: 2359
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 2199
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2108
type: A, layer: 1, pos: 304
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 3531
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 431
type: A, layer: 1, pos: 3491
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 313
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 110
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 3220
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 2404
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 2976
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2187
type: A, layer: 1, pos: 2286
type: A, layer: 1, pos: 3285
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 2086
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 3267
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 2293
type: A, layer: 1, pos: 2292
type: A, layer: 1, pos: 3296
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 2492
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 2491
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 3238
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2506
type: A, layer: 1, pos: 2041
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2073
type: A, layer: 1, pos: 2610
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 2625
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 2085
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2204
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2687
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 318

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0292833, upper bound: 0.0294397
time: 68.38 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0294389, upper bound: 0.0294412
time: 39.05 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 114.08 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 114.08
Output dim: 3, lower bound: -0.0289528, upper bound: 0.0294519
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 114.08
Output dim: 3, lower bound: -0.0291052, upper bound: 0.0294372
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 114.08
Output dim: 3, lower bound: -0.0290706, upper bound: 0.0294425
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 114.08
Output dim: 3, lower bound: -0.0292262, upper bound: 0.0294428
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 114.08
Output dim: 3, lower bound: -0.0291616, upper bound: 0.0294519
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 114.08
Output dim: 3, lower bound: -0.0293184, upper bound: 0.0294546
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 114.08
Output dim: 3, lower bound: -0.0292833, upper bound: 0.0294397
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 114.08
Output dim: 3, lower bound: -0.0294389, upper bound: 0.0294412

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -3.0442276, -1.4669526, -3.0449514, -1.4659886, -0.5168339, 0.5165653
1: -2.2270222, -0.5700269, -2.2313075, -0.5649786, -0.5748745, 0.5737617
2: -2.5361779, -1.9034861, -2.5382051, -1.9019853, -0.1338019, 0.1342998
3: -0.8532782, -0.5444196, -0.8543925, -0.5432978, -0.0705930, 0.0705405
4: -2.3981347, -1.4631526, -2.3987141, -1.4627104, -0.1991068, 0.1994029
5: -0.4018213, -0.1606520, -0.4027417, -0.1597480, -0.0520220, 0.0519988
6: -1.2608415, -0.6765720, -1.2624301, -0.6750963, -0.1443194, 0.1445151
7: -0.5155355, 0.2308503, -0.5156162, 0.2307588, -0.1283740, 0.1286344
8: -3.5465121, -1.8124132, -3.5473146, -1.8101435, -0.6267869, 0.6262934
9: -4.2367697, -2.6074440, -4.2388744, -2.6045268, -0.5864009, 0.5855649

Time for backsubstitution: 6.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2426
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 3477
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 2635
type: B, layer: 1, pos: 3498
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2541
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 3545
type: B, layer: 1, pos: 2391
type: B, layer: 1, pos: 2211
type: B, layer: 1, pos: 2365
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 3041
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 2539
type: B, layer: 1, pos: 2210
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 2231
type: B, layer: 1, pos: 2199
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 328
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2108
type: B, layer: 1, pos: 304
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 318
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2117
type: B, layer: 1, pos: 2203
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 3531
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 431
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 313
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 110
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 3220
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 2404
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 2557
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 3462
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 2087
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2976
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2187
type: B, layer: 1, pos: 3285
type: B, layer: 1, pos: 2286
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2086
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 2600
type: B, layer: 1, pos: 3267
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 2293
type: B, layer: 1, pos: 2292
type: B, layer: 1, pos: 3296
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2052
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 2492
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 2491
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 2190
type: B, layer: 1, pos: 3238
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2041
type: B, layer: 1, pos: 2506
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2073
type: B, layer: 1, pos: 2610
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2056
type: B, layer: 1, pos: 2051
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 2625
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 2085
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2069
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2204
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2234
type: B, layer: 1, pos: 2243
type: B, layer: 1, pos: 2687
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 3149

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2426

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0289066, upper bound: 0.0293821
time: 89.83 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0289112, upper bound: 0.0294027
time: 63.49 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -3.0443406, -1.4668951, -3.0450246, -1.4659448, -0.5170135, 0.5166952
1: -2.2297010, -0.5580344, -2.2313156, -0.5557179, -0.5866601, 0.5719917
2: -2.5385153, -1.9027719, -2.5399976, -1.9019639, -0.1336822, 0.1368679
3: -0.8551439, -0.5436978, -0.8560630, -0.5432128, -0.0700453, 0.0729801
4: -2.4008653, -1.4627419, -2.4008455, -1.4627028, -0.1991659, 0.2020059
5: -0.4031123, -0.1602023, -0.4040029, -0.1597382, -0.0513743, 0.0536896
6: -1.2653433, -0.6753012, -1.2661724, -0.6750350, -0.1432595, 0.1495267
7: -0.5169845, 0.2307727, -0.5167236, 0.2307588, -0.1283750, 0.1302282
8: -3.5462170, -1.8091049, -3.5473280, -1.8075972, -0.6313484, 0.6255673
9: -4.2379541, -2.6000471, -4.2388754, -2.5987992, -0.5939865, 0.5855228

Time for backsubstitution: 5.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2426
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 3477
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 2635
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 3498
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2541
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 3545
type: B, layer: 1, pos: 2391
type: B, layer: 1, pos: 2211
type: B, layer: 1, pos: 2365
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 3041
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 2539
type: B, layer: 1, pos: 2210
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 2231
type: B, layer: 1, pos: 2199
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 328
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 318
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2108
type: B, layer: 1, pos: 304
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2117
type: B, layer: 1, pos: 2203
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 3531
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 431
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 313
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 110
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 3220
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 2404
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 2557
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 3462
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 2087
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2976
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2187
type: B, layer: 1, pos: 2286
type: B, layer: 1, pos: 3285
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 2086
type: B, layer: 1, pos: 2600
type: B, layer: 1, pos: 3267
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 2293
type: B, layer: 1, pos: 2292
type: B, layer: 1, pos: 3296
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2052
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 2492
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 2491
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 2190
type: B, layer: 1, pos: 3238
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2506
type: B, layer: 1, pos: 2041
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2073
type: B, layer: 1, pos: 2610
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2056
type: B, layer: 1, pos: 2051
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 2625
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 2085
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2069
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2204
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2234
type: B, layer: 1, pos: 2243
type: B, layer: 1, pos: 2687
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 3149

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2426

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0290637, upper bound: 0.0293813
time: 28.23 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0290671, upper bound: 0.0294039
time: 101.39 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3.0445611, -1.4669526, -3.0457077, -1.4654889, -0.5177749, 0.5173806
1: -2.2276263, -0.5700269, -2.2323165, -0.5626369, -0.5773963, 0.5743886
2: -2.5362020, -1.9016521, -2.5403125, -1.8994185, -0.1353250, 0.1383930
3: -0.8532781, -0.5436065, -0.8552767, -0.5421469, -0.0712035, 0.0720834
4: -2.3984060, -1.4630694, -2.3991146, -1.4620967, -0.1999973, 0.1998442
5: -0.4018214, -0.1599968, -0.4035820, -0.1588873, -0.0523633, 0.0531985
6: -1.2608417, -0.6757760, -1.2631330, -0.6738851, -0.1451812, 0.1457456
7: -0.5155424, 0.2311519, -0.5166525, 0.2312266, -0.1286780, 0.1303501
8: -3.5468097, -1.8124132, -3.5477955, -1.8094349, -0.6278579, 0.6266561
9: -4.2369504, -2.6073065, -4.2391047, -2.6035972, -0.5874187, 0.5859917

Time for backsubstitution: 5.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2426
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 3477
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 2635
type: B, layer: 1, pos: 3498
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2541
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 3545
type: B, layer: 1, pos: 2391
type: B, layer: 1, pos: 2365
type: B, layer: 1, pos: 2211
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 3041
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 2539
type: B, layer: 1, pos: 2210
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 2231
type: B, layer: 1, pos: 2199
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 328
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2108
type: B, layer: 1, pos: 304
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 318
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2117
type: B, layer: 1, pos: 2203
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 3531
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 431
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 313
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 110
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 3220
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 2404
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 2557
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 3462
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 2087
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2976
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2187
type: B, layer: 1, pos: 3285
type: B, layer: 1, pos: 2286
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 2086
type: B, layer: 1, pos: 2600
type: B, layer: 1, pos: 3267
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 2293
type: B, layer: 1, pos: 2292
type: B, layer: 1, pos: 3296
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2052
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 2492
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 2491
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 2190
type: B, layer: 1, pos: 3238
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2506
type: B, layer: 1, pos: 2041
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2073
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2610
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2056
type: B, layer: 1, pos: 2051
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 2625
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 2085
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2069
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2204
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2234
type: B, layer: 1, pos: 2243
type: B, layer: 1, pos: 2687
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 3149

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2426

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0290285, upper bound: 0.0293826
time: 20.76 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0290320, upper bound: 0.0293969
time: 26.39 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3.0446739, -1.4668951, -3.0457819, -1.4654458, -0.5179545, 0.5175096
1: -2.2303057, -0.5580344, -2.2323241, -0.5533762, -0.5891818, 0.5726185
2: -2.5385389, -1.9009379, -2.5421042, -1.8993973, -0.1352053, 0.1409597
3: -0.8551439, -0.5428846, -0.8569471, -0.5420616, -0.0706558, 0.0745228
4: -2.4011364, -1.4626592, -2.4012456, -1.4620891, -0.2000565, 0.2024465
5: -0.4031124, -0.1595470, -0.4048432, -0.1588775, -0.0517156, 0.0548893
6: -1.2653433, -0.6745050, -1.2668750, -0.6738234, -0.1441216, 0.1507569
7: -0.5169918, 0.2310746, -0.5177600, 0.2312266, -0.1286789, 0.1319436
8: -3.5465145, -1.8091044, -3.5478086, -1.8068893, -0.6324195, 0.6259300
9: -4.2381358, -2.5999095, -4.2391052, -2.5978682, -0.5950030, 0.5859498

Time for backsubstitution: 6.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2426
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 3477
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 2635
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 3498
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2541
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 3545
type: B, layer: 1, pos: 2391
type: B, layer: 1, pos: 2365
type: B, layer: 1, pos: 2211
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 3041
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 2539
type: B, layer: 1, pos: 2210
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 2231
type: B, layer: 1, pos: 2199
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 328
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 318
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2108
type: B, layer: 1, pos: 304
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2117
type: B, layer: 1, pos: 2203
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 3531
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 431
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 313
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 110
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 3220
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 2404
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 2557
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 3462
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 2087
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2976
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2187
type: B, layer: 1, pos: 2286
type: B, layer: 1, pos: 3285
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 2086
type: B, layer: 1, pos: 2600
type: B, layer: 1, pos: 3267
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 2293
type: B, layer: 1, pos: 2292
type: B, layer: 1, pos: 3296
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2052
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 2492
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 2491
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 2190
type: B, layer: 1, pos: 3238
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2506
type: B, layer: 1, pos: 2041
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2073
type: B, layer: 1, pos: 2610
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2056
type: B, layer: 1, pos: 2051
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 2625
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 2085
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2069
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2204
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2234
type: B, layer: 1, pos: 2243
type: B, layer: 1, pos: 2687
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 3149

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2426

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0291849, upper bound: 0.0293851
time: 11.17 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0291887, upper bound: 0.0294102
time: 6.25 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -3.0452206, -1.4633417, -3.0451128, -1.4633243, -0.5206249, 0.5164237
1: -2.2289689, -0.5612845, -2.2313116, -0.5585408, -0.5832404, 0.5723597
2: -2.5417235, -1.9021423, -2.5422633, -1.9018953, -0.1331296, 0.1397108
3: -0.8563129, -0.5436577, -0.8565120, -0.5432194, -0.0703718, 0.0734621
4: -2.3983195, -1.4629388, -2.3988557, -1.4625518, -0.1996099, 0.1995026
5: -0.4047193, -0.1599983, -0.4047529, -0.1597193, -0.0517884, 0.0546922
6: -1.2632937, -0.6760819, -1.2640589, -0.6750910, -0.1442826, 0.1466497
7: -0.5157688, 0.2311352, -0.5161410, 0.2310115, -0.1295099, 0.1295585
8: -3.5478091, -1.8067880, -3.5473511, -1.8060005, -0.6322311, 0.6254382
9: -4.2378035, -2.6026843, -4.2388744, -2.6010413, -0.5908819, 0.5848899

Time for backsubstitution: 6.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2426
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 3477
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 2635
type: B, layer: 1, pos: 3498
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2541
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 3545
type: B, layer: 1, pos: 2391
type: B, layer: 1, pos: 2211
type: B, layer: 1, pos: 2365
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 3041
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 2539
type: B, layer: 1, pos: 2210
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 2231
type: B, layer: 1, pos: 2199
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 328
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2108
type: B, layer: 1, pos: 304
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 318
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2117
type: B, layer: 1, pos: 2203
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 3531
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 431
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 313
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 110
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 3220
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 2404
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 2557
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 3462
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 2087
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2976
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2187
type: B, layer: 1, pos: 3285
type: B, layer: 1, pos: 2286
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2086
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 2600
type: B, layer: 1, pos: 3267
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 2293
type: B, layer: 1, pos: 2292
type: B, layer: 1, pos: 3296
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2052
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 2492
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 2491
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 2190
type: B, layer: 1, pos: 3238
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2506
type: B, layer: 1, pos: 2041
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2073
type: B, layer: 1, pos: 2610
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2056
type: B, layer: 1, pos: 2051
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 2625
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 2085
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2069
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2204
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2234
type: B, layer: 1, pos: 2243
type: B, layer: 1, pos: 2687
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 3149

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2426

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0291196, upper bound: 0.0293845
time: 157.08 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0291241, upper bound: 0.0294040
time: 55.58 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -3.0453327, -1.4632843, -3.0451865, -1.4632800, -0.5208058, 0.5165518
1: -2.2316480, -0.5492921, -2.2313190, -0.5492802, -0.5950261, 0.5705895
2: -2.5440612, -1.9014269, -2.5440555, -1.9018734, -0.1330095, 0.1422794
3: -0.8581812, -0.5429353, -0.8581824, -0.5431340, -0.0698243, 0.0759031
4: -2.4010482, -1.4625285, -2.4009864, -1.4625444, -0.1996679, 0.2021031
5: -0.4060130, -0.1595484, -0.4060141, -0.1597095, -0.0511413, 0.0563845
6: -1.2677976, -0.6748110, -1.2678010, -0.6750295, -0.1432266, 0.1516603
7: -0.5172174, 0.2310576, -0.5172483, 0.2310115, -0.1295103, 0.1311519
8: -3.5475171, -1.8034792, -3.5473657, -1.8034546, -0.6367943, 0.6247110
9: -4.2389889, -2.5952899, -4.2388754, -2.5953131, -0.5984671, 0.5848456

Time for backsubstitution: 6.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2426
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 3477
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 2635
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 3498
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2541
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 3545
type: B, layer: 1, pos: 2391
type: B, layer: 1, pos: 2365
type: B, layer: 1, pos: 2211
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 3041
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 2539
type: B, layer: 1, pos: 2210
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 2231
type: B, layer: 1, pos: 2199
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 328
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 318
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2108
type: B, layer: 1, pos: 304
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2117
type: B, layer: 1, pos: 2203
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 3531
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 431
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 313
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 110
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 3220
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 2404
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 2557
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 3462
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 2087
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2976
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2187
type: B, layer: 1, pos: 2286
type: B, layer: 1, pos: 3285
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 2086
type: B, layer: 1, pos: 2600
type: B, layer: 1, pos: 3267
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 2293
type: B, layer: 1, pos: 2292
type: B, layer: 1, pos: 3296
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2052
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 2492
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 2491
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 2190
type: B, layer: 1, pos: 3238
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2506
type: B, layer: 1, pos: 2041
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2073
type: B, layer: 1, pos: 2610
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2056
type: B, layer: 1, pos: 2051
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 2625
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 2085
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2069
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2204
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2234
type: B, layer: 1, pos: 2243
type: B, layer: 1, pos: 2687
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 3149

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 2426

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0292750, upper bound: 0.0293784
time: 47.30 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0292794, upper bound: 0.0294048
time: 207.96 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3.0455546, -1.4633417, -3.0458691, -1.4628248, -0.5215661, 0.5172411
1: -2.2295728, -0.5612845, -2.2323203, -0.5561996, -0.5857621, 0.5729866
2: -2.5417473, -1.9003083, -2.5443709, -1.8993286, -0.1346525, 0.1438047
3: -0.8563130, -0.5428447, -0.8573961, -0.5420688, -0.0709825, 0.0750046
4: -2.3985906, -1.4628561, -2.3992558, -1.4619386, -0.2004995, 0.1999455
5: -0.4047195, -0.1593431, -0.4055930, -0.1588586, -0.0521300, 0.0558919
6: -1.2632942, -0.6752863, -1.2647619, -0.6738796, -0.1451444, 0.1478803
7: -0.5157764, 0.2314368, -0.5171759, 0.2314792, -0.1298145, 0.1312819
8: -3.5481076, -1.8067882, -3.5478327, -1.8052919, -0.6333022, 0.6258007
9: -4.2379847, -2.6025476, -4.2391047, -2.6001110, -0.5918997, 0.5853167

Time for backsubstitution: 6.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2426
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 3477
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 2635
type: B, layer: 1, pos: 3498
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2541
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 3545
type: B, layer: 1, pos: 2391
type: B, layer: 1, pos: 2365
type: B, layer: 1, pos: 2211
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 3041
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 2539
type: B, layer: 1, pos: 2210
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 2231
type: B, layer: 1, pos: 2199
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 328
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2108
type: B, layer: 1, pos: 304
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 318
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2117
type: B, layer: 1, pos: 2203
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 3531
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 431
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 313
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 110
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 3220
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 2404
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 2557
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 3462
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 2087
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2976
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2187
type: B, layer: 1, pos: 3285
type: B, layer: 1, pos: 2286
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 2086
type: B, layer: 1, pos: 2600
type: B, layer: 1, pos: 3267
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 2293
type: B, layer: 1, pos: 2292
type: B, layer: 1, pos: 3296
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2052
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 2492
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 2491
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 2190
type: B, layer: 1, pos: 3238
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2506
type: B, layer: 1, pos: 2041
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2073
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2610
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2056
type: B, layer: 1, pos: 2051
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 2625
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 2085
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2069
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2204
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2234
type: B, layer: 1, pos: 2243
type: B, layer: 1, pos: 2687
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 3149

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2426

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0292408, upper bound: 0.0293804
time: 21.12 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0292441, upper bound: 0.0294140
time: 9.65 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3.0456674, -1.4632843, -3.0459428, -1.4627810, -0.5217468, 0.5173681
1: -2.2322528, -0.5492921, -2.2323275, -0.5469379, -0.5975478, 0.5712163
2: -2.5440853, -1.8995929, -2.5461626, -1.8993068, -0.1345325, 0.1463720
3: -0.8581812, -0.5421222, -0.8590664, -0.5419831, -0.0704345, 0.0774454
4: -2.4013195, -1.4624454, -2.4013860, -1.4619308, -0.2005575, 0.2025456
5: -0.4060130, -0.1588931, -0.4068543, -0.1588487, -0.0514828, 0.0575842
6: -1.2677981, -0.6740153, -1.2685034, -0.6738180, -0.1440887, 0.1528905
7: -0.5172249, 0.2313595, -0.5182835, 0.2314792, -0.1298149, 0.1328750
8: -3.5478139, -1.8034792, -3.5478463, -1.8027468, -0.6378655, 0.6250736
9: -4.2391701, -2.5951519, -4.2391052, -2.5943828, -0.5994837, 0.5852726

Time for backsubstitution: 6.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2426
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 3477
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 2635
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 3498
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2541
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 3545
type: B, layer: 1, pos: 2391
type: B, layer: 1, pos: 2365
type: B, layer: 1, pos: 2211
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 3041
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 2539
type: B, layer: 1, pos: 2210
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 2231
type: B, layer: 1, pos: 2199
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 328
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 318
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2108
type: B, layer: 1, pos: 304
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2117
type: B, layer: 1, pos: 2203
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 3531
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 431
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 313
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 110
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 3220
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 2404
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 2557
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 3462
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 2087
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2976
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2187
type: B, layer: 1, pos: 2286
type: B, layer: 1, pos: 3285
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 2086
type: B, layer: 1, pos: 2600
type: B, layer: 1, pos: 3267
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 2293
type: B, layer: 1, pos: 2292
type: B, layer: 1, pos: 3296
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2052
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 2492
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 2491
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 2190
type: B, layer: 1, pos: 3238
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2506
type: B, layer: 1, pos: 2041
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2073
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 2610
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2056
type: B, layer: 1, pos: 2051
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 2625
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 2085
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2069
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2204
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2234
type: B, layer: 1, pos: 2243
type: B, layer: 1, pos: 2687
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 3149

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2426

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0293967, upper bound: 0.0293814
time: 51.47 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0294004, upper bound: 0.0294049
time: 29.79 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 87.64 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 87.64
Output dim: 3, lower bound: -0.0289066, upper bound: 0.0293821
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 87.64
Output dim: 3, lower bound: -0.0289112, upper bound: 0.0294027
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 87.64
Output dim: 3, lower bound: -0.0290637, upper bound: 0.0293813
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 87.64
Output dim: 3, lower bound: -0.0290671, upper bound: 0.0294039
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 87.64
Output dim: 3, lower bound: -0.0290285, upper bound: 0.0293826
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 87.64
Output dim: 3, lower bound: -0.0290320, upper bound: 0.0293969
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 87.64
Output dim: 3, lower bound: -0.0291849, upper bound: 0.0293851
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 87.64
Output dim: 3, lower bound: -0.0291887, upper bound: 0.0294102
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 87.64
Output dim: 3, lower bound: -0.0291196, upper bound: 0.0293845
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 87.64
Output dim: 3, lower bound: -0.0291241, upper bound: 0.0294040
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 87.64
Output dim: 3, lower bound: -0.0292750, upper bound: 0.0293784
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 87.64
Output dim: 3, lower bound: -0.0292794, upper bound: 0.0294048
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 87.64
Output dim: 3, lower bound: -0.0292408, upper bound: 0.0293804
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 87.64
Output dim: 3, lower bound: -0.0292441, upper bound: 0.0294140
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 87.64
Output dim: 3, lower bound: -0.0293967, upper bound: 0.0293814
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 87.64
Output dim: 3, lower bound: -0.0294004, upper bound: 0.0294049

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -3.0442126, -1.4687791, -3.0449333, -1.4682920, -0.5142519, 0.5144500
1: -2.2270222, -0.5734944, -2.2313075, -0.5693526, -0.5709772, 0.5705855
2: -2.5354738, -1.9034867, -2.5373187, -1.9019856, -0.1330179, 0.1333465
3: -0.8526495, -0.5444230, -0.8536205, -0.5433019, -0.0699109, 0.0697086
4: -2.3968341, -1.4631526, -2.3970776, -1.4627106, -0.1978023, 0.1978226
5: -0.4013374, -0.1606525, -0.4021425, -0.1597486, -0.0514264, 0.0512723
6: -1.2602053, -0.6765721, -1.2616446, -0.6750968, -0.1437203, 0.1437860
7: -0.5150307, 0.2308503, -0.5149884, 0.2307588, -0.1278506, 0.1280040
8: -3.5465119, -1.8175585, -3.5473142, -1.8166337, -0.6215199, 0.6219798
9: -4.2367697, -2.6121562, -4.2388744, -2.6103616, -0.5815274, 0.5815582

Time for backsubstitution: 6.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 3477
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 2635
type: A, layer: 1, pos: 3498
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 2125
type: A, layer: 1, pos: 3545
type: A, layer: 1, pos: 2391
type: A, layer: 1, pos: 2211
type: A, layer: 1, pos: 2365
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 3041
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2210
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 2359
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 2199
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2108
type: A, layer: 1, pos: 304
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 3531
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 2426
type: A, layer: 1, pos: 3491
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 431
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 313
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 110
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 3220
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 2404
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 2976
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2187
type: A, layer: 1, pos: 3285
type: A, layer: 1, pos: 2286
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2086
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 3267
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 2293
type: A, layer: 1, pos: 2292
type: A, layer: 1, pos: 3296
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 2492
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 2491
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 3238
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2041
type: A, layer: 1, pos: 2506
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 2073
type: A, layer: 1, pos: 2610
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 2625
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 2085
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2204
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2687
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 2410

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0288570, upper bound: 0.0293378
time: 49.82 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0288652, upper bound: 0.0293374
time: 139.45 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -3.0442204, -1.4690719, -3.0500827, -1.4682732, -0.5146218, 0.5223034
1: -2.2270222, -0.5725589, -2.2404265, -0.5678473, -0.5717109, 0.5826688
2: -2.5351892, -1.9034868, -2.5370686, -1.9006791, -0.1356973, 0.1334256
3: -0.8525345, -0.5444221, -0.8535284, -0.5427427, -0.0710799, 0.0697679
4: -2.3972292, -1.4631526, -2.3976181, -1.4593942, -0.2023671, 0.1980869
5: -0.4011236, -0.1606524, -0.4019426, -0.1595670, -0.0521520, 0.0513090
6: -1.2602479, -0.6765729, -1.2617673, -0.6748360, -0.1440324, 0.1438488
7: -0.5150438, 0.2308503, -0.5150565, 0.2317537, -0.1295900, 0.1280963
8: -3.5465117, -1.8156574, -3.5602753, -1.8139169, -0.6225690, 0.6384906
9: -4.2367697, -2.6103983, -4.2506366, -2.6079366, -0.5824348, 0.5966579

Time for backsubstitution: 5.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 3477
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 2635
type: A, layer: 1, pos: 3498
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 2125
type: A, layer: 1, pos: 3545
type: A, layer: 1, pos: 2391
type: A, layer: 1, pos: 2211
type: A, layer: 1, pos: 2365
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 3041
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2210
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 2359
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 2199
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2108
type: A, layer: 1, pos: 304
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 3531
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 3491
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 431
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 2426
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 313
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 110
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 3220
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 2404
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 2976
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2187
type: A, layer: 1, pos: 3285
type: A, layer: 1, pos: 2286
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2086
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 3267
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 2293
type: A, layer: 1, pos: 2292
type: A, layer: 1, pos: 3296
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 2492
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 2491
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 3238
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2041
type: A, layer: 1, pos: 2506
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 2073
type: A, layer: 1, pos: 2610
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 2625
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 2085
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2204
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2687
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 2410

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0288610, upper bound: 0.0293628
time: 15.99 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0288662, upper bound: 0.0293667
time: 5.95 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -3.0443261, -1.4687214, -3.0450072, -1.4682486, -0.5144317, 0.5145801
1: -2.2297013, -0.5615015, -2.2313147, -0.5600910, -0.5827626, 0.5688156
2: -2.5378106, -1.9027723, -2.5391109, -1.9019645, -0.1328983, 0.1359151
3: -0.8545161, -0.5437011, -0.8552919, -0.5432167, -0.0693646, 0.0721483
4: -2.3995647, -1.4627419, -2.3992085, -1.4627028, -0.1978612, 0.2004255
5: -0.4026291, -0.1602028, -0.4034048, -0.1597388, -0.0507829, 0.0529632
6: -1.2647089, -0.6753013, -1.2653887, -0.6750352, -0.1426636, 0.1487976
7: -0.5164797, 0.2307727, -0.5160958, 0.2307588, -0.1278515, 0.1295979
8: -3.5462170, -1.8142502, -3.5473280, -1.8140881, -0.6260815, 0.6212536
9: -4.2379541, -2.6047595, -4.2388754, -2.6046336, -0.5891126, 0.5815158

Time for backsubstitution: 5.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 3477
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 2635
type: A, layer: 1, pos: 3498
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 2125
type: A, layer: 1, pos: 3545
type: A, layer: 1, pos: 2391
type: A, layer: 1, pos: 2211
type: A, layer: 1, pos: 2365
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 3041
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2210
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 2359
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 2199
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2108
type: A, layer: 1, pos: 304
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 3531
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 2426
type: A, layer: 1, pos: 3491
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 431
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 313
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 110
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 3220
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 2404
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 2976
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2187
type: A, layer: 1, pos: 3285
type: A, layer: 1, pos: 2286
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2086
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 3267
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 2293
type: A, layer: 1, pos: 2292
type: A, layer: 1, pos: 3296
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 2492
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 2491
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 3238
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2041
type: A, layer: 1, pos: 2506
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 2610
type: A, layer: 1, pos: 2073
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 2625
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 2085
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2204
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2687
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2410

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0290132, upper bound: 0.0293353
time: 131.10 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0290190, upper bound: 0.0293336
time: 106.92 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -3.0443339, -1.4690144, -3.0501556, -1.4682300, -0.5148014, 0.5224333
1: -2.2297013, -0.5605655, -2.2404337, -0.5585847, -0.5834965, 0.5808986
2: -2.5375266, -1.9027727, -2.5388610, -1.9006575, -0.1355781, 0.1359941
3: -0.8544013, -0.5437003, -0.8551996, -0.5426575, -0.0705327, 0.0722077
4: -2.3999596, -1.4627419, -2.3997495, -1.4593866, -0.2024263, 0.2006898
5: -0.4024157, -0.1602027, -0.4032048, -0.1595572, -0.0515080, 0.0529998
6: -1.2647513, -0.6753019, -1.2655104, -0.6747750, -0.1429747, 0.1488604
7: -0.5164927, 0.2307727, -0.5161638, 0.2317537, -0.1295909, 0.1296903
8: -3.5462172, -1.8123481, -3.5602896, -1.8113704, -0.6271305, 0.6377645
9: -4.2379541, -2.6030018, -4.2506380, -2.6022081, -0.5900201, 0.5966164

Time for backsubstitution: 6.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 3477
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 2635
type: A, layer: 1, pos: 3498
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 2125
type: A, layer: 1, pos: 3545
type: A, layer: 1, pos: 2391
type: A, layer: 1, pos: 2211
type: A, layer: 1, pos: 2365
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 3041
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2210
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 2359
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 2199
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2108
type: A, layer: 1, pos: 304
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 3531
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 3491
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 431
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 2426
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 313
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 110
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 3220
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 2404
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 2976
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2187
type: A, layer: 1, pos: 3285
type: A, layer: 1, pos: 2286
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2086
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 3267
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 2293
type: A, layer: 1, pos: 2292
type: A, layer: 1, pos: 3296
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 2492
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 2491
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 3238
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2041
type: A, layer: 1, pos: 2506
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 2610
type: A, layer: 1, pos: 2073
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 2625
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 2085
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2204
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2687
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2410

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0290168, upper bound: 0.0293327
time: 194.32 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0290224, upper bound: 0.0293601
time: 54.16 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -3.0445464, -1.4687791, -3.0456903, -1.4677927, -0.5151930, 0.5152650
1: -2.2276258, -0.5734944, -2.2323165, -0.5670109, -0.5734990, 0.5712124
2: -2.5354974, -1.9016531, -2.5394254, -1.8994191, -0.1345409, 0.1374393
3: -0.8526495, -0.5436097, -0.8545046, -0.5421509, -0.0705214, 0.0712516
4: -2.3971057, -1.4630699, -2.3974783, -1.4620967, -0.1986929, 0.1982636
5: -0.4013374, -0.1599972, -0.4029827, -0.1588879, -0.0517678, 0.0524719
6: -1.2602054, -0.6757762, -1.2623475, -0.6738855, -0.1445821, 0.1450164
7: -0.5150379, 0.2311519, -0.5160247, 0.2312266, -0.1281545, 0.1297190
8: -3.5468092, -1.8175585, -3.5477958, -1.8159251, -0.6225910, 0.6223424
9: -4.2369504, -2.6120193, -4.2391047, -2.6094313, -0.5825450, 0.5819851

Time for backsubstitution: 6.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 3477
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 2635
type: A, layer: 1, pos: 3498
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 2125
type: A, layer: 1, pos: 3545
type: A, layer: 1, pos: 2391
type: A, layer: 1, pos: 2211
type: A, layer: 1, pos: 2365
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 3041
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2210
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 2359
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 2199
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2108
type: A, layer: 1, pos: 304
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 3531
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 2426
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 431
type: A, layer: 1, pos: 3491
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 313
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 110
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 3220
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 2404
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 2976
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2187
type: A, layer: 1, pos: 3285
type: A, layer: 1, pos: 2286
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2086
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 3267
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 2293
type: A, layer: 1, pos: 2292
type: A, layer: 1, pos: 3296
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 2492
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 2491
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 3238
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2041
type: A, layer: 1, pos: 2506
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 2073
type: A, layer: 1, pos: 2610
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 2625
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 2085
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2204
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2687
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2410

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0289778, upper bound: 0.0293303
time: 36.93 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0289840, upper bound: 0.0293300
time: 50.79 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -3.0445540, -1.4690719, -3.0508380, -1.4677737, -0.5155628, 0.5231171
1: -2.2276263, -0.5725589, -2.2414353, -0.5655046, -0.5742327, 0.5832957
2: -2.5352132, -1.9016529, -2.5391762, -1.8981125, -0.1372204, 0.1375184
3: -0.8525346, -0.5436091, -0.8544124, -0.5415917, -0.0716902, 0.0713109
4: -2.3975005, -1.4630699, -2.3980188, -1.4587808, -0.2032578, 0.1985278
5: -0.4011237, -0.1599972, -0.4027829, -0.1587063, -0.0524933, 0.0525086
6: -1.2602479, -0.6757768, -1.2624698, -0.6736249, -0.1448942, 0.1450792
7: -0.5150507, 0.2311519, -0.5160930, 0.2322216, -0.1298939, 0.1298110
8: -3.5468092, -1.8156574, -3.5607562, -1.8132081, -0.6236401, 0.6388531
9: -4.2369504, -2.6102617, -4.2508664, -2.6070061, -0.5834525, 0.5970848

Time for backsubstitution: 6.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 3477
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 2635
type: A, layer: 1, pos: 3498
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 2125
type: A, layer: 1, pos: 3545
type: A, layer: 1, pos: 2391
type: A, layer: 1, pos: 2211
type: A, layer: 1, pos: 2365
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 3041
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2210
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 2359
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 2199
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2108
type: A, layer: 1, pos: 304
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 3531
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 431
type: A, layer: 1, pos: 3491
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 2426
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 313
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 110
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 3220
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 2404
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 2976
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2187
type: A, layer: 1, pos: 3285
type: A, layer: 1, pos: 2286
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2086
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 3267
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 2293
type: A, layer: 1, pos: 2292
type: A, layer: 1, pos: 3296
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 2492
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 2491
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 3238
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2041
type: A, layer: 1, pos: 2506
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 2073
type: A, layer: 1, pos: 2610
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 2625
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 2085
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2204
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2687
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2410

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0289822, upper bound: 0.0293649
time: 8.75 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0289871, upper bound: 0.0293572
time: 166.20 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -3.0446596, -1.4687214, -3.0457635, -1.4677491, -0.5153725, 0.5153944
1: -2.2303054, -0.5615015, -2.2323239, -0.5577488, -0.5852844, 0.5694424
2: -2.5378349, -1.9009383, -2.5412178, -1.8993976, -0.1344215, 0.1400064
3: -0.8545162, -0.5428880, -0.8561760, -0.5420655, -0.0699751, 0.0736911
4: -2.3998358, -1.4626591, -2.3996089, -1.4620888, -0.1987518, 0.2008658
5: -0.4026291, -0.1595475, -0.4042450, -0.1588780, -0.0511242, 0.0541627
6: -1.2647091, -0.6745054, -1.2660911, -0.6738237, -0.1435257, 0.1500277
7: -0.5164871, 0.2310746, -0.5171322, 0.2312266, -0.1281555, 0.1313126
8: -3.5465140, -1.8142502, -3.5478091, -1.8133793, -0.6271524, 0.6216163
9: -4.2381358, -2.6046216, -4.2391052, -2.6037025, -0.5901291, 0.5819428

Time for backsubstitution: 6.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 3477
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 2635
type: A, layer: 1, pos: 3498
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 2125
type: A, layer: 1, pos: 3545
type: A, layer: 1, pos: 2391
type: A, layer: 1, pos: 2211
type: A, layer: 1, pos: 2365
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 3041
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2210
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 2359
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 2199
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2108
type: A, layer: 1, pos: 304
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 3531
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 2426
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 3491
type: A, layer: 1, pos: 431
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 313
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 110
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 3220
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 2404
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 2976
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2187
type: A, layer: 1, pos: 3285
type: A, layer: 1, pos: 2286
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2086
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 3267
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 2293
type: A, layer: 1, pos: 2292
type: A, layer: 1, pos: 3296
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 2492
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 2491
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 3238
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2041
type: A, layer: 1, pos: 2506
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 2610
type: A, layer: 1, pos: 2073
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 2625
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 2085
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2204
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2687
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2410

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0291346, upper bound: 0.0293323
time: 235.68 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0291398, upper bound: 0.0293298
time: 116.24 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -3.0446663, -1.4690144, -3.0509124, -1.4677310, -0.5157424, 0.5232460
1: -2.2303057, -0.5605655, -2.2414432, -0.5562434, -0.5860181, 0.5815256
2: -2.5375512, -1.9009386, -2.5409682, -1.8980908, -0.1371013, 0.1400854
3: -0.8544012, -0.5428873, -0.8560836, -0.5415063, -0.0711430, 0.0737504
4: -2.4002304, -1.4626589, -2.4001498, -1.4587729, -0.2033170, 0.2011301
5: -0.4024158, -0.1595474, -0.4040450, -0.1586964, -0.0518493, 0.0541994
6: -1.2647517, -0.6745063, -1.2662134, -0.6735632, -0.1438368, 0.1500905
7: -0.5164999, 0.2310746, -0.5172001, 0.2322216, -0.1298949, 0.1314045
8: -3.5465145, -1.8123486, -3.5607700, -1.8106620, -0.6282016, 0.6381269
9: -4.2381358, -2.6028638, -4.2508678, -2.6012776, -0.5910366, 0.5970436

Time for backsubstitution: 6.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 3477
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 2635
type: A, layer: 1, pos: 3498
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 2125
type: A, layer: 1, pos: 3545
type: A, layer: 1, pos: 2391
type: A, layer: 1, pos: 2211
type: A, layer: 1, pos: 2365
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 3041
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2210
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 2359
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 2199
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2108
type: A, layer: 1, pos: 304
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 3531
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 3491
type: A, layer: 1, pos: 431
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 2426
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 313
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 110
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 3220
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 2404
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 2976
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2187
type: A, layer: 1, pos: 3285
type: A, layer: 1, pos: 2286
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2086
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 3267
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 2293
type: A, layer: 1, pos: 2292
type: A, layer: 1, pos: 3296
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 2492
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 2491
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 3238
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2041
type: A, layer: 1, pos: 2506
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 2610
type: A, layer: 1, pos: 2073
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 2625
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 2085
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2204
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2687
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2410

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0291383, upper bound: 0.0293532
time: 165.32 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0291437, upper bound: 0.0293667
time: 5.93 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -3.0452058, -1.4651682, -3.0450954, -1.4656274, -0.5180432, 0.5143076
1: -2.2289679, -0.5647511, -2.2313118, -0.5629148, -0.5793431, 0.5691835
2: -2.5410194, -1.9021424, -2.5413768, -1.9018958, -0.1323457, 0.1387573
3: -0.8556870, -0.5436609, -0.8557422, -0.5432235, -0.0696911, 0.0726300
4: -2.3970184, -1.4629389, -2.3972194, -1.4625522, -0.1983051, 0.1979217
5: -0.4042384, -0.1599988, -0.4041561, -0.1597199, -0.0511944, 0.0539656
6: -1.2626607, -0.6760825, -1.2632768, -0.6750915, -0.1436869, 0.1459205
7: -0.5152643, 0.2311352, -0.5155131, 0.2310115, -0.1289865, 0.1289268
8: -3.5478096, -1.8119333, -3.5473509, -1.8124905, -0.6269642, 0.6211245
9: -4.2378035, -2.6073971, -4.2388744, -2.6068747, -0.5860083, 0.5808834

Time for backsubstitution: 6.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 3477
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 2635
type: A, layer: 1, pos: 3498
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 2125
type: A, layer: 1, pos: 3545
type: A, layer: 1, pos: 2391
type: A, layer: 1, pos: 2211
type: A, layer: 1, pos: 2365
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 3041
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 2210
type: A, layer: 1, pos: 2359
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 2199
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2108
type: A, layer: 1, pos: 304
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 3531
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 2426
type: A, layer: 1, pos: 3491
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 431
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 313
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 110
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 3220
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 2404
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 2976
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2187
type: A, layer: 1, pos: 3285
type: A, layer: 1, pos: 2286
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2086
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 3267
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 2293
type: A, layer: 1, pos: 2292
type: A, layer: 1, pos: 3296
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 2492
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 2491
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 3238
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2506
type: A, layer: 1, pos: 2041
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2073
type: A, layer: 1, pos: 2610
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 2625
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 2085
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2204
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2687
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2410

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0290696, upper bound: 0.0293457
time: 9.61 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0290750, upper bound: 0.0293389
time: 27.63 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -3.0452135, -1.4654610, -3.0502443, -1.4656091, -0.5184132, 0.5221598
1: -2.2289684, -0.5638161, -2.2404301, -0.5614090, -0.5800767, 0.5812666
2: -2.5407357, -1.9021428, -2.5411270, -1.9005891, -0.1350249, 0.1388363
3: -0.8555720, -0.5436603, -0.8556498, -0.5426642, -0.0708601, 0.0726893
4: -2.3974133, -1.4629390, -2.3977604, -1.4592359, -0.2028702, 0.1981861
5: -0.4040243, -0.1599987, -0.4039560, -0.1595384, -0.0519201, 0.0540023
6: -1.2627039, -0.6760831, -1.2633990, -0.6748310, -0.1439995, 0.1459832
7: -0.5152774, 0.2311352, -0.5155813, 0.2320064, -0.1307265, 0.1290190
8: -3.5478091, -1.8100317, -3.5603123, -1.8097734, -0.6280134, 0.6376351
9: -4.2378035, -2.6056392, -4.2506366, -2.6044505, -0.5869157, 0.5959831

Time for backsubstitution: 6.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 3477
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 2635
type: A, layer: 1, pos: 3498
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 2125
type: A, layer: 1, pos: 3545
type: A, layer: 1, pos: 2391
type: A, layer: 1, pos: 2211
type: A, layer: 1, pos: 2365
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 3041
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 2210
type: A, layer: 1, pos: 2359
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 2199
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2108
type: A, layer: 1, pos: 304
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 3531
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 3491
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 431
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 2426
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 313
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 110
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 3220
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 2404
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 2976
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2187
type: A, layer: 1, pos: 3285
type: A, layer: 1, pos: 2286
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2086
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 3267
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 2293
type: A, layer: 1, pos: 2292
type: A, layer: 1, pos: 3296
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 2492
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 2491
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 3238
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2506
type: A, layer: 1, pos: 2041
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 2073
type: A, layer: 1, pos: 2610
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 2625
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 2085
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2204
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2687
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 2410

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0290732, upper bound: 0.0293718
time: 6.36 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0290788, upper bound: 0.0293644
time: 83.17 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 45.18 + 3579.36 = 3624.54 seconds
