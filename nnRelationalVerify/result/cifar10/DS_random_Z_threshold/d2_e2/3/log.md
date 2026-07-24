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
execution time: IAR + RelationalAnalysis = 7.76 + 37.51 = 45.27 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0294424, upper bound: 0.0294492

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3220
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 516
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 3531
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 431
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 856
type: DSZ, layer: 1, pos: 821

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2204

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0294429, upper bound: 0.0294479
time: 135.39 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0294429, upper bound: 0.0294454
time: 43.06 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 178.47 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 178.47
Output dim: 3, lower bound: -0.0294429, upper bound: 0.0294479
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 178.47
Output dim: 3, lower bound: -0.0294429, upper bound: 0.0294454

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -3.0459356, -1.4632723, -3.0459356, -1.4632723, -0.5215368, 0.5215368
1: -2.2329955, -0.5492516, -2.2329955, -0.5492516, -0.5965176, 0.5965177
2: -2.5440857, -1.8995905, -2.5440857, -1.8995905, -0.1443744, 0.1443744
3: -0.8581853, -0.5421180, -0.8581853, -0.5421180, -0.0767322, 0.0767322
4: -2.4013999, -1.4624283, -2.4013999, -1.4624283, -0.2031510, 0.2031510
5: -0.4060171, -0.1588888, -0.4060171, -0.1588888, -0.0569329, 0.0569329
6: -1.2678086, -0.6738276, -1.2678086, -0.6738276, -0.1527211, 0.1527211
7: -0.5173369, 0.2315183, -0.5173369, 0.2315183, -0.1320915, 0.1320915
8: -3.5481367, -1.8033185, -3.5481367, -1.8033185, -0.6374951, 0.6374950
9: -4.2394500, -2.5951118, -4.2394500, -2.5951118, -0.5994052, 0.5994052

Time for backsubstitution: 5.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 516
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 3220
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 856
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 3531
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 431
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 2072

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 860

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0294188, upper bound: 0.0294499
time: 8.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0294359, upper bound: 0.0294213
time: 54.74 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -3.0459356, -1.4632723, -3.0459356, -1.4632723, -0.5215368, 0.5215368
1: -2.2329955, -0.5492516, -2.2329955, -0.5492516, -0.5965176, 0.5965177
2: -2.5440857, -1.8995905, -2.5440857, -1.8995905, -0.1443744, 0.1443744
3: -0.8581853, -0.5421180, -0.8581853, -0.5421180, -0.0767322, 0.0767322
4: -2.4013999, -1.4624283, -2.4013999, -1.4624283, -0.2031510, 0.2031510
5: -0.4060171, -0.1588888, -0.4060171, -0.1588888, -0.0569329, 0.0569329
6: -1.2678086, -0.6738276, -1.2678086, -0.6738276, -0.1527211, 0.1527211
7: -0.5173369, 0.2315183, -0.5173369, 0.2315183, -0.1320915, 0.1320915
8: -3.5481367, -1.8033185, -3.5481367, -1.8033185, -0.6374951, 0.6374950
9: -4.2394500, -2.5951118, -4.2394500, -2.5951118, -0.5994052, 0.5994052

Time for backsubstitution: 5.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 856
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 3531
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 516
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3220
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 431
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 696

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 864

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0294153, upper bound: 0.0294504
time: 9.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0294367, upper bound: 0.0294223
time: 19.52 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 34.24 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 34.24
Output dim: 3, lower bound: -0.0294188, upper bound: 0.0294499
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 34.24
Output dim: 3, lower bound: -0.0294359, upper bound: 0.0294213
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 34.24
Output dim: 3, lower bound: -0.0294153, upper bound: 0.0294504
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 34.24
Output dim: 3, lower bound: -0.0294367, upper bound: 0.0294223

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.0459356, -1.4632723, -3.0459356, -1.4632723, -0.5213751, 0.5213507
1: -2.2329955, -0.5492516, -2.2329955, -0.5492516, -0.5960562, 0.5959862
2: -2.5440857, -1.8995905, -2.5440857, -1.8995905, -0.1441891, 0.1442136
3: -0.8581853, -0.5421180, -0.8581853, -0.5421180, -0.0766709, 0.0766788
4: -2.4013999, -1.4624283, -2.4013999, -1.4624283, -0.2027756, 0.2028400
5: -0.4060171, -0.1588888, -0.4060171, -0.1588888, -0.0568928, 0.0568979
6: -1.2678086, -0.6738276, -1.2678086, -0.6738276, -0.1527149, 0.1527155
7: -0.5173369, 0.2315183, -0.5173369, 0.2315183, -0.1319066, 0.1319483
8: -3.5481367, -1.8033185, -3.5481367, -1.8033185, -0.6369422, 0.6368048
9: -4.2394500, -2.5951118, -4.2394500, -2.5951118, -0.5987654, 0.5986690

Time for backsubstitution: 5.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3531
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 516
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 3220
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 856
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 431
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 203

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2228

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0293643, upper bound: 0.0294358
time: 12.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0294070, upper bound: 0.0293963
time: 7.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.0459356, -1.4632723, -3.0459356, -1.4632723, -0.5213507, 0.5213751
1: -2.2329955, -0.5492516, -2.2329955, -0.5492516, -0.5959863, 0.5960561
2: -2.5440857, -1.8995905, -2.5440857, -1.8995905, -0.1442136, 0.1441891
3: -0.8581853, -0.5421180, -0.8581853, -0.5421180, -0.0766788, 0.0766709
4: -2.4013999, -1.4624283, -2.4013999, -1.4624283, -0.2028400, 0.2027757
5: -0.4060171, -0.1588888, -0.4060171, -0.1588888, -0.0568979, 0.0568928
6: -1.2678086, -0.6738276, -1.2678086, -0.6738276, -0.1527155, 0.1527149
7: -0.5173369, 0.2315183, -0.5173369, 0.2315183, -0.1319483, 0.1319066
8: -3.5481367, -1.8033185, -3.5481367, -1.8033185, -0.6368049, 0.6369421
9: -4.2394500, -2.5951118, -4.2394500, -2.5951118, -0.5986691, 0.5987653

Time for backsubstitution: 5.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 856
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 3220
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 516
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 3531
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 431
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 735

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2191

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0294353, upper bound: 0.0294274
time: 132.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0294356, upper bound: 0.0294322
time: 8.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.0459356, -1.4632723, -3.0459356, -1.4632723, -0.5200428, 0.5200031
1: -2.2329955, -0.5492516, -2.2329955, -0.5492516, -0.5928894, 0.5927943
2: -2.5440857, -1.8995905, -2.5440857, -1.8995905, -0.1431621, 0.1431940
3: -0.8581853, -0.5421180, -0.8581853, -0.5421180, -0.0762461, 0.0762588
4: -2.4013999, -1.4624283, -2.4013999, -1.4624283, -0.2001065, 0.2002167
5: -0.4060171, -0.1588888, -0.4060171, -0.1588888, -0.0566367, 0.0566442
6: -1.2678086, -0.6738276, -1.2678086, -0.6738276, -0.1526134, 0.1526180
7: -0.5173369, 0.2315183, -0.5173369, 0.2315183, -0.1314217, 0.1314466
8: -3.5481367, -1.8033185, -3.5481367, -1.8033185, -0.6334308, 0.6332778
9: -4.2394500, -2.5951118, -4.2394500, -2.5951118, -0.5947781, 0.5946425

Time for backsubstitution: 5.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 3220
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3531
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 516
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 431
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 856
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 875

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0293977, upper bound: 0.0294429
time: 13.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0294142, upper bound: 0.0294205
time: 344.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.0459356, -1.4632723, -3.0459356, -1.4632723, -0.5200032, 0.5200428
1: -2.2329955, -0.5492516, -2.2329955, -0.5492516, -0.5927943, 0.5928893
2: -2.5440857, -1.8995905, -2.5440857, -1.8995905, -0.1431940, 0.1431621
3: -0.8581853, -0.5421180, -0.8581853, -0.5421180, -0.0762588, 0.0762461
4: -2.4013999, -1.4624283, -2.4013999, -1.4624283, -0.2002167, 0.2001065
5: -0.4060171, -0.1588888, -0.4060171, -0.1588888, -0.0566442, 0.0566367
6: -1.2678086, -0.6738276, -1.2678086, -0.6738276, -0.1526180, 0.1526134
7: -0.5173369, 0.2315183, -0.5173369, 0.2315183, -0.1314466, 0.1314217
8: -3.5481367, -1.8033185, -3.5481367, -1.8033185, -0.6332778, 0.6334308
9: -4.2394500, -2.5951118, -4.2394500, -2.5951118, -0.5946425, 0.5947781

Time for backsubstitution: 5.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 516
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 856
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 3531
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 431
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 3220
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2034

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2073

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0294301, upper bound: 0.0294246
time: 7.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0294301, upper bound: 0.0294240
time: 7.65 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 20.82 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 20.82
Output dim: 3, lower bound: -0.0293643, upper bound: 0.0294358
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 20.82
Output dim: 3, lower bound: -0.0294070, upper bound: 0.0293963
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 20.82
Output dim: 3, lower bound: -0.0294353, upper bound: 0.0294274
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 20.82
Output dim: 3, lower bound: -0.0294356, upper bound: 0.0294322
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 20.82
Output dim: 3, lower bound: -0.0293977, upper bound: 0.0294429
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 20.82
Output dim: 3, lower bound: -0.0294142, upper bound: 0.0294205
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 20.82
Output dim: 3, lower bound: -0.0294301, upper bound: 0.0294246
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 20.82
Output dim: 3, lower bound: -0.0294301, upper bound: 0.0294240

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.0459356, -1.4632723, -3.0459356, -1.4632723, -0.5213624, 0.5211734
1: -2.2329955, -0.5492516, -2.2329955, -0.5492516, -0.5960215, 0.5955573
2: -2.5440857, -1.8995905, -2.5440857, -1.8995905, -0.1440549, 0.1441967
3: -0.8581853, -0.5421180, -0.8581853, -0.5421180, -0.0766162, 0.0766726
4: -2.4013999, -1.4624283, -2.4013999, -1.4624283, -0.2024136, 0.2028145
5: -0.4060171, -0.1588888, -0.4060171, -0.1588888, -0.0568595, 0.0568939
6: -1.2678086, -0.6738276, -1.2678086, -0.6738276, -0.1527036, 0.1527137
7: -0.5173369, 0.2315183, -0.5173369, 0.2315183, -0.1318225, 0.1319294
8: -3.5481367, -1.8033185, -3.5481367, -1.8033185, -0.6369063, 0.6363428
9: -4.2394500, -2.5951118, -4.2394500, -2.5951118, -0.5987297, 0.5981373

Time for backsubstitution: 5.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 856
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 3531
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 516
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 3220
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 431
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 2034

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2031

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0293641, upper bound: 0.0294262
time: 40.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0293640, upper bound: 0.0294091
time: 381.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.0459356, -1.4632723, -3.0459356, -1.4632723, -0.5211979, 0.5213380
1: -2.2329955, -0.5492516, -2.2329955, -0.5492516, -0.5956272, 0.5959517
2: -2.5440857, -1.8995905, -2.5440857, -1.8995905, -0.1441722, 0.1440794
3: -0.8581853, -0.5421180, -0.8581853, -0.5421180, -0.0766646, 0.0766241
4: -2.4013999, -1.4624283, -2.4013999, -1.4624283, -0.2027501, 0.2024780
5: -0.4060171, -0.1588888, -0.4060171, -0.1588888, -0.0568888, 0.0568646
6: -1.2678086, -0.6738276, -1.2678086, -0.6738276, -0.1527131, 0.1527042
7: -0.5173369, 0.2315183, -0.5173369, 0.2315183, -0.1318877, 0.1318642
8: -3.5481367, -1.8033185, -3.5481367, -1.8033185, -0.6364802, 0.6367689
9: -4.2394500, -2.5951118, -4.2394500, -2.5951118, -0.5982337, 0.5986333

Time for backsubstitution: 6.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 3220
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 856
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 431
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 3531
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 516
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 101

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2077

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0294001, upper bound: 0.0293796
time: 28.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0294016, upper bound: 0.0293848
time: 36.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.0459356, -1.4632723, -3.0459356, -1.4632723, -0.5190853, 0.5189868
1: -2.2329955, -0.5492516, -2.2329955, -0.5492516, -0.5944099, 0.5944003
2: -2.5440857, -1.8995905, -2.5440857, -1.8995905, -0.1440462, 0.1440287
3: -0.8581853, -0.5421180, -0.8581853, -0.5421180, -0.0766741, 0.0766663
4: -2.4013999, -1.4624283, -2.4013999, -1.4624283, -0.2025644, 0.2025036
5: -0.4060171, -0.1588888, -0.4060171, -0.1588888, -0.0568017, 0.0568008
6: -1.2678086, -0.6738276, -1.2678086, -0.6738276, -0.1522720, 0.1522890
7: -0.5173369, 0.2315183, -0.5173369, 0.2315183, -0.1307186, 0.1307346
8: -3.5481367, -1.8033185, -3.5481367, -1.8033185, -0.6306368, 0.6304474
9: -4.2394500, -2.5951118, -4.2394500, -2.5951118, -0.5971490, 0.5971725

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 431
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 516
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 3220
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 856
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 3531
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 809

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2071

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0294356, upper bound: 0.0294292
time: 10.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0294356, upper bound: 0.0294235
time: 258.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.0459356, -1.4632723, -3.0459356, -1.4632723, -0.5189624, 0.5191097
1: -2.2329955, -0.5492516, -2.2329955, -0.5492516, -0.5943305, 0.5944797
2: -2.5440857, -1.8995905, -2.5440857, -1.8995905, -0.1440533, 0.1440216
3: -0.8581853, -0.5421180, -0.8581853, -0.5421180, -0.0766742, 0.0766662
4: -2.4013999, -1.4624283, -2.4013999, -1.4624283, -0.2025679, 0.2025000
5: -0.4060171, -0.1588888, -0.4060171, -0.1588888, -0.0568059, 0.0567966
6: -1.2678086, -0.6738276, -1.2678086, -0.6738276, -0.1522897, 0.1522713
7: -0.5173369, 0.2315183, -0.5173369, 0.2315183, -0.1307763, 0.1306769
8: -3.5481367, -1.8033185, -3.5481367, -1.8033185, -0.6303101, 0.6307741
9: -4.2394500, -2.5951118, -4.2394500, -2.5951118, -0.5970762, 0.5972453

Time for backsubstitution: 6.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 431
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 516
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 3220
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 856
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3531
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2051

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 110

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0293332, upper bound: 0.0293557
time: 81.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0293733, upper bound: 0.0293183
time: 67.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.0459356, -1.4632723, -3.0459356, -1.4632723, -0.5197771, 0.5197171
1: -2.2329955, -0.5492516, -2.2329955, -0.5492516, -0.5921485, 0.5919781
2: -2.5440857, -1.8995905, -2.5440857, -1.8995905, -0.1428872, 0.1429484
3: -0.8581853, -0.5421180, -0.8581853, -0.5421180, -0.0761534, 0.0761713
4: -2.4013999, -1.4624283, -2.4013999, -1.4624283, -0.1995380, 0.1996998
5: -0.4060171, -0.1588888, -0.4060171, -0.1588888, -0.0565771, 0.0565871
6: -1.2678086, -0.6738276, -1.2678086, -0.6738276, -0.1526038, 0.1526089
7: -0.5173369, 0.2315183, -0.5173369, 0.2315183, -0.1311437, 0.1311986
8: -3.5481367, -1.8033185, -3.5481367, -1.8033185, -0.6324893, 0.6322173
9: -4.2394500, -2.5951118, -4.2394500, -2.5951118, -0.5937570, 0.5935157

Time for backsubstitution: 6.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 516
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 431
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 856
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 3531
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 3220
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 785

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0293683, upper bound: 0.0294444
time: 11.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0293943, upper bound: 0.0293906
time: 228.96 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.0459356, -1.4632723, -3.0459356, -1.4632723, -0.5197568, 0.5197374
1: -2.2329955, -0.5492516, -2.2329955, -0.5492516, -0.5920732, 0.5920534
2: -2.5440857, -1.8995905, -2.5440857, -1.8995905, -0.1429165, 0.1429190
3: -0.8581853, -0.5421180, -0.8581853, -0.5421180, -0.0761586, 0.0761661
4: -2.4013999, -1.4624283, -2.4013999, -1.4624283, -0.1995896, 0.1996482
5: -0.4060171, -0.1588888, -0.4060171, -0.1588888, -0.0565795, 0.0565847
6: -1.2678086, -0.6738276, -1.2678086, -0.6738276, -0.1526043, 0.1526083
7: -0.5173369, 0.2315183, -0.5173369, 0.2315183, -0.1311737, 0.1311685
8: -3.5481367, -1.8033185, -3.5481367, -1.8033185, -0.6323702, 0.6323363
9: -4.2394500, -2.5951118, -4.2394500, -2.5951118, -0.5936513, 0.5936214

Time for backsubstitution: 6.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 3220
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 431
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 3531
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 516
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 856
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2059

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 885

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0294142, upper bound: 0.0294239
time: 96.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0294142, upper bound: 0.0294226
time: 34.68 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.0459356, -1.4632723, -3.0459356, -1.4632723, -0.5199886, 0.5200008
1: -2.2329955, -0.5492516, -2.2329955, -0.5492516, -0.5927889, 0.5928620
2: -2.5440857, -1.8995905, -2.5440857, -1.8995905, -0.1431866, 0.1431583
3: -0.8581853, -0.5421180, -0.8581853, -0.5421180, -0.0762587, 0.0762460
4: -2.4013999, -1.4624283, -2.4013999, -1.4624283, -0.2002133, 0.2000812
5: -0.4060171, -0.1588888, -0.4060171, -0.1588888, -0.0566426, 0.0566350
6: -1.2678086, -0.6738276, -1.2678086, -0.6738276, -0.1526069, 0.1526062
7: -0.5173369, 0.2315183, -0.5173369, 0.2315183, -0.1314457, 0.1314199
8: -3.5481367, -1.8033185, -3.5481367, -1.8033185, -0.6332821, 0.6333551
9: -4.2394500, -2.5951118, -4.2394500, -2.5951118, -0.5946347, 0.5947782

Time for backsubstitution: 6.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 516
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 856
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 431
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 3531
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 3220
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2125

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 886

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0294298, upper bound: 0.0294158
time: 160.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0294298, upper bound: 0.0294097
time: 270.10 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.0459356, -1.4632723, -3.0459356, -1.4632723, -0.5199611, 0.5200428
1: -2.2329955, -0.5492516, -2.2329955, -0.5492516, -0.5927669, 0.5928893
2: -2.5440857, -1.8995905, -2.5440857, -1.8995905, -0.1431901, 0.1431621
3: -0.8581853, -0.5421180, -0.8581853, -0.5421180, -0.0762587, 0.0762461
4: -2.4013999, -1.4624283, -2.4013999, -1.4624283, -0.2001913, 0.2001065
5: -0.4060171, -0.1588888, -0.4060171, -0.1588888, -0.0566442, 0.0566350
6: -1.2678086, -0.6738276, -1.2678086, -0.6738276, -0.1526180, 0.1526023
7: -0.5173369, 0.2315183, -0.5173369, 0.2315183, -0.1314448, 0.1314217
8: -3.5481367, -1.8033185, -3.5481367, -1.8033185, -0.6332022, 0.6334308
9: -4.2394500, -2.5951118, -4.2394500, -2.5951118, -0.5946425, 0.5947702

Time for backsubstitution: 6.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 3531
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 431
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 856
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 516
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 3220
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 892

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 867

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0294236, upper bound: 0.0294081
time: 351.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0294292, upper bound: 0.0294058
time: 236.81 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 594.65 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 594.65
Output dim: 3, lower bound: -0.0293641, upper bound: 0.0294262
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 594.65
Output dim: 3, lower bound: -0.0293640, upper bound: 0.0294091
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 594.65
Output dim: 3, lower bound: -0.0294001, upper bound: 0.0293796
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 594.65
Output dim: 3, lower bound: -0.0294016, upper bound: 0.0293848
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 594.65
Output dim: 3, lower bound: -0.0294356, upper bound: 0.0294292
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 594.65
Output dim: 3, lower bound: -0.0294356, upper bound: 0.0294235
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 594.65
Output dim: 3, lower bound: -0.0293332, upper bound: 0.0293557
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 594.65
Output dim: 3, lower bound: -0.0293733, upper bound: 0.0293183
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 594.65
Output dim: 3, lower bound: -0.0293683, upper bound: 0.0294444
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 594.65
Output dim: 3, lower bound: -0.0293943, upper bound: 0.0293906
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 594.65
Output dim: 3, lower bound: -0.0294142, upper bound: 0.0294239
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 594.65
Output dim: 3, lower bound: -0.0294142, upper bound: 0.0294226
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 594.65
Output dim: 3, lower bound: -0.0294298, upper bound: 0.0294158
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 594.65
Output dim: 3, lower bound: -0.0294298, upper bound: 0.0294097
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 594.65
Output dim: 3, lower bound: -0.0294236, upper bound: 0.0294081
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 594.65
Output dim: 3, lower bound: -0.0294292, upper bound: 0.0294058

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.0459356, -1.4632723, -3.0459356, -1.4632723, -0.5213579, 0.5211740
1: -2.2329955, -0.5492516, -2.2329955, -0.5492516, -0.5960324, 0.5955415
2: -2.5440857, -1.8995905, -2.5440857, -1.8995905, -0.1440546, 0.1441958
3: -0.8581853, -0.5421180, -0.8581853, -0.5421180, -0.0766162, 0.0766725
4: -2.4013999, -1.4624283, -2.4013999, -1.4624283, -0.2024048, 0.2028100
5: -0.4060171, -0.1588888, -0.4060171, -0.1588888, -0.0568596, 0.0568939
6: -1.2678086, -0.6738276, -1.2678086, -0.6738276, -0.1526973, 0.1527166
7: -0.5173369, 0.2315183, -0.5173369, 0.2315183, -0.1318215, 0.1319290
8: -3.5481367, -1.8033185, -3.5481367, -1.8033185, -0.6369077, 0.6363381
9: -4.2394500, -2.5951118, -4.2394500, -2.5951118, -0.5987443, 0.5981132

Time for backsubstitution: 5.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 431
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 856
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 3220
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 3531
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 516
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2199

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2610

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0293492, upper bound: 0.0294240
time: 7.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0293492, upper bound: 0.0294240
time: 7.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.0459356, -1.4632723, -3.0459356, -1.4632723, -0.5213624, 0.5211689
1: -2.2329955, -0.5492516, -2.2329955, -0.5492516, -0.5960056, 0.5955573
2: -2.5440857, -1.8995905, -2.5440857, -1.8995905, -0.1440549, 0.1441964
3: -0.8581853, -0.5421180, -0.8581853, -0.5421180, -0.0766161, 0.0766726
4: -2.4013999, -1.4624283, -2.4013999, -1.4624283, -0.2024136, 0.2028056
5: -0.4060171, -0.1588888, -0.4060171, -0.1588888, -0.0568594, 0.0568939
6: -1.2678086, -0.6738276, -1.2678086, -0.6738276, -0.1527036, 0.1527075
7: -0.5173369, 0.2315183, -0.5173369, 0.2315183, -0.1318220, 0.1319294
8: -3.5481367, -1.8033185, -3.5481367, -1.8033185, -0.6369015, 0.6363428
9: -4.2394500, -2.5951118, -4.2394500, -2.5951118, -0.5987055, 0.5981373

Time for backsubstitution: 6.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 3531
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 856
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 431
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 516
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 3220
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2086

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0293633, upper bound: 0.0294269
time: 37.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0293633, upper bound: 0.0294331
time: 13.89 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.0459356, -1.4632723, -3.0459356, -1.4632723, -0.5210184, 0.5211500
1: -2.2329955, -0.5492516, -2.2329955, -0.5492516, -0.5946822, 0.5949011
2: -2.5440857, -1.8995905, -2.5440857, -1.8995905, -0.1438069, 0.1437415
3: -0.8581853, -0.5421180, -0.8581853, -0.5421180, -0.0765925, 0.0765544
4: -2.4013999, -1.4624283, -2.4013999, -1.4624283, -0.2024152, 0.2021665
5: -0.4060171, -0.1588888, -0.4060171, -0.1588888, -0.0568451, 0.0568221
6: -1.2678086, -0.6738276, -1.2678086, -0.6738276, -0.1523032, 0.1523123
7: -0.5173369, 0.2315183, -0.5173369, 0.2315183, -0.1317763, 0.1317529
8: -3.5481367, -1.8033185, -3.5481367, -1.8033185, -0.6363220, 0.6365933
9: -4.2394500, -2.5951118, -4.2394500, -2.5951118, -0.5971982, 0.5974973

Time for backsubstitution: 6.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 431
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 3220
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 3531
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 516
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 856
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2583

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2410

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0293501, upper bound: 0.0293482
time: 6.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0293553, upper bound: 0.0293274
time: 172.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.0459356, -1.4632723, -3.0459356, -1.4632723, -0.5210098, 0.5211587
1: -2.2329955, -0.5492516, -2.2329955, -0.5492516, -0.5945767, 0.5950066
2: -2.5440857, -1.8995905, -2.5440857, -1.8995905, -0.1438342, 0.1437142
3: -0.8581853, -0.5421180, -0.8581853, -0.5421180, -0.0765950, 0.0765519
4: -2.4013999, -1.4624283, -2.4013999, -1.4624283, -0.2024386, 0.2021431
5: -0.4060171, -0.1588888, -0.4060171, -0.1588888, -0.0568463, 0.0568209
6: -1.2678086, -0.6738276, -1.2678086, -0.6738276, -0.1523211, 0.1522943
7: -0.5173369, 0.2315183, -0.5173369, 0.2315183, -0.1317765, 0.1317528
8: -3.5481367, -1.8033185, -3.5481367, -1.8033185, -0.6363045, 0.6366107
9: -4.2394500, -2.5951118, -4.2394500, -2.5951118, -0.5970976, 0.5975978

Time for backsubstitution: 6.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 431
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 516
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 3220
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 856
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 3531
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2410

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0293969, upper bound: 0.0293855
time: 13.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0294014, upper bound: 0.0293834
time: 6.37 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.0459356, -1.4632723, -3.0459356, -1.4632723, -0.5179392, 0.5177721
1: -2.2329955, -0.5492516, -2.2329955, -0.5492516, -0.5942459, 0.5942276
2: -2.5440857, -1.8995905, -2.5440857, -1.8995905, -0.1439823, 0.1439640
3: -0.8581853, -0.5421180, -0.8581853, -0.5421180, -0.0766738, 0.0766658
4: -2.4013999, -1.4624283, -2.4013999, -1.4624283, -0.2020978, 0.2020084
5: -0.4060171, -0.1588888, -0.4060171, -0.1588888, -0.0567612, 0.0567615
6: -1.2678086, -0.6738276, -1.2678086, -0.6738276, -0.1519548, 0.1519890
7: -0.5173369, 0.2315183, -0.5173369, 0.2315183, -0.1307046, 0.1307200
8: -3.5481367, -1.8033185, -3.5481367, -1.8033185, -0.6290781, 0.6287764
9: -4.2394500, -2.5951118, -4.2394500, -2.5951118, -0.5959930, 0.5960838

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 3220
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 516
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 856
type: DSZ, layer: 1, pos: 431
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 3531
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 736

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3267

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0294154, upper bound: 0.0294210
time: 167.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0294357, upper bound: 0.0294030
time: 17.91 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 190.96 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 190.96
Output dim: 3, lower bound: -0.0293492, upper bound: 0.0294240
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 190.96
Output dim: 3, lower bound: -0.0293492, upper bound: 0.0294240
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 190.96
Output dim: 3, lower bound: -0.0293633, upper bound: 0.0294269
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 190.96
Output dim: 3, lower bound: -0.0293633, upper bound: 0.0294331
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 190.96
Output dim: 3, lower bound: -0.0293501, upper bound: 0.0293482
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 190.96
Output dim: 3, lower bound: -0.0293553, upper bound: 0.0293274
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 190.96
Output dim: 3, lower bound: -0.0293969, upper bound: 0.0293855
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 190.96
Output dim: 3, lower bound: -0.0294014, upper bound: 0.0293834
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 190.96
Output dim: 3, lower bound: -0.0294154, upper bound: 0.0294210
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 190.96
Output dim: 3, lower bound: -0.0294357, upper bound: 0.0294030
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 190.96
Output dim: 3, lower bound: -0.0294356, upper bound: 0.0294235
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 190.96
Output dim: 3, lower bound: -0.0293332, upper bound: 0.0293557
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 190.96
Output dim: 3, lower bound: -0.0293733, upper bound: 0.0293183
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 190.96
Output dim: 3, lower bound: -0.0293683, upper bound: 0.0294444
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 190.96
Output dim: 3, lower bound: -0.0293943, upper bound: 0.0293906
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 190.96
Output dim: 3, lower bound: -0.0294142, upper bound: 0.0294239
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 190.96
Output dim: 3, lower bound: -0.0294142, upper bound: 0.0294226
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 190.96
Output dim: 3, lower bound: -0.0294298, upper bound: 0.0294158
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 190.96
Output dim: 3, lower bound: -0.0294298, upper bound: 0.0294097
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 190.96
Output dim: 3, lower bound: -0.0294236, upper bound: 0.0294081
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 190.96
Output dim: 3, lower bound: -0.0294292, upper bound: 0.0294058

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 45.27 + 3661.05 = 3706.32 seconds
