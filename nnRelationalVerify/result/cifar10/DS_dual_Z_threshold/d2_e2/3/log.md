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
execution time: IAR + RelationalAnalysis = 8.39 + 36.88 = 45.28 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0294424, upper bound: 0.0294492

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 516
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 431
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 856
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3220
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 3531

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2374

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0294197, upper bound: 0.0294429
time: 10.67 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0294279, upper bound: 0.0294240
time: 60.99 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 71.75 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 71.75
Output dim: 3, lower bound: -0.0294197, upper bound: 0.0294429
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 71.75
Output dim: 3, lower bound: -0.0294279, upper bound: 0.0294240

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -3.0459356, -1.4632723, -3.0459356, -1.4632723, -0.5167571, 0.5163734
1: -2.2329955, -0.5492516, -2.2329955, -0.5492516, -0.5899903, 0.5895088
2: -2.5440857, -1.8995905, -2.5440857, -1.8995905, -0.1443114, 0.1443096
3: -0.8581853, -0.5421180, -0.8581853, -0.5421180, -0.0766245, 0.0766337
4: -2.4013999, -1.4624283, -2.4013999, -1.4624283, -0.2028738, 0.2028750
5: -0.4060171, -0.1588888, -0.4060171, -0.1588888, -0.0566692, 0.0566869
6: -1.2678086, -0.6738276, -1.2678086, -0.6738276, -0.1527045, 0.1527143
7: -0.5173369, 0.2315183, -0.5173369, 0.2315183, -0.1313135, 0.1313089
8: -3.5481367, -1.8033185, -3.5481367, -1.8033185, -0.6317819, 0.6314055
9: -4.2394500, -2.5951118, -4.2394500, -2.5951118, -0.5924246, 0.5919678

Time for backsubstitution: 6.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 516
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 431
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 856
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3220
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 3531

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2600

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0293758, upper bound: 0.0293871
time: 172.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0293758, upper bound: 0.0293782
time: 287.01 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -3.0459356, -1.4632723, -3.0459356, -1.4632723, -0.5163734, 0.5167570
1: -2.2329955, -0.5492516, -2.2329955, -0.5492516, -0.5895088, 0.5899903
2: -2.5440857, -1.8995905, -2.5440857, -1.8995905, -0.1443095, 0.1443114
3: -0.8581853, -0.5421180, -0.8581853, -0.5421180, -0.0766337, 0.0766245
4: -2.4013999, -1.4624283, -2.4013999, -1.4624283, -0.2028750, 0.2028738
5: -0.4060171, -0.1588888, -0.4060171, -0.1588888, -0.0566869, 0.0566692
6: -1.2678086, -0.6738276, -1.2678086, -0.6738276, -0.1527143, 0.1527045
7: -0.5173369, 0.2315183, -0.5173369, 0.2315183, -0.1313088, 0.1313135
8: -3.5481367, -1.8033185, -3.5481367, -1.8033185, -0.6314054, 0.6317819
9: -4.2394500, -2.5951118, -4.2394500, -2.5951118, -0.5919678, 0.5924245

Time for backsubstitution: 6.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 516
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 431
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 856
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3220
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 3531

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2600

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0293842, upper bound: 0.0293804
time: 61.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0293842, upper bound: 0.0293809
time: 264.01 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 332.11 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 332.11
Output dim: 3, lower bound: -0.0293758, upper bound: 0.0293871
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 332.11
Output dim: 3, lower bound: -0.0293758, upper bound: 0.0293782
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 332.11
Output dim: 3, lower bound: -0.0293842, upper bound: 0.0293804
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 332.11
Output dim: 3, lower bound: -0.0293842, upper bound: 0.0293809

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.0459356, -1.4632723, -3.0459356, -1.4632723, -0.5167283, 0.5164488
1: -2.2329955, -0.5492516, -2.2329955, -0.5492516, -0.5899131, 0.5898268
2: -2.5440857, -1.8995905, -2.5440857, -1.8995905, -0.1443102, 0.1443043
3: -0.8581853, -0.5421180, -0.8581853, -0.5421180, -0.0766214, 0.0766307
4: -2.4013999, -1.4624283, -2.4013999, -1.4624283, -0.2028965, 0.2028663
5: -0.4060171, -0.1588888, -0.4060171, -0.1588888, -0.0566666, 0.0566835
6: -1.2678086, -0.6738276, -1.2678086, -0.6738276, -0.1527047, 0.1527140
7: -0.5173369, 0.2315183, -0.5173369, 0.2315183, -0.1313278, 0.1312938
8: -3.5481367, -1.8033185, -3.5481367, -1.8033185, -0.6316900, 0.6316670
9: -4.2394500, -2.5951118, -4.2394500, -2.5951118, -0.5923283, 0.5922906

Time for backsubstitution: 6.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 516
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 431
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 856
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3220
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 3531

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2391

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0293188, upper bound: 0.0293846
time: 33.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0293731, upper bound: 0.0293428
time: 6.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.0459356, -1.4632723, -3.0459356, -1.4632723, -0.5167571, 0.5163447
1: -2.2329955, -0.5492516, -2.2329955, -0.5492516, -0.5899903, 0.5894316
2: -2.5440857, -1.8995905, -2.5440857, -1.8995905, -0.1443062, 0.1443096
3: -0.8581853, -0.5421180, -0.8581853, -0.5421180, -0.0766215, 0.0766337
4: -2.4013999, -1.4624283, -2.4013999, -1.4624283, -0.2028651, 0.2028750
5: -0.4060171, -0.1588888, -0.4060171, -0.1588888, -0.0566658, 0.0566869
6: -1.2678086, -0.6738276, -1.2678086, -0.6738276, -0.1527042, 0.1527143
7: -0.5173369, 0.2315183, -0.5173369, 0.2315183, -0.1312985, 0.1313089
8: -3.5481367, -1.8033185, -3.5481367, -1.8033185, -0.6317819, 0.6313135
9: -4.2394500, -2.5951118, -4.2394500, -2.5951118, -0.5924246, 0.5918716

Time for backsubstitution: 6.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 516
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 431
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 856
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3220
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 3531

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2391

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0293188, upper bound: 0.0293847
time: 32.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0293731, upper bound: 0.0293428
time: 6.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.0459356, -1.4632723, -3.0459356, -1.4632723, -0.5163447, 0.5168276
1: -2.2329955, -0.5492516, -2.2329955, -0.5492516, -0.5894316, 0.5903164
2: -2.5440857, -1.8995905, -2.5440857, -1.8995905, -0.1443083, 0.1443062
3: -0.8581853, -0.5421180, -0.8581853, -0.5421180, -0.0766311, 0.0766215
4: -2.4013999, -1.4624283, -2.4013999, -1.4624283, -0.2028977, 0.2028651
5: -0.4060171, -0.1588888, -0.4060171, -0.1588888, -0.0566845, 0.0566658
6: -1.2678086, -0.6738276, -1.2678086, -0.6738276, -0.1527145, 0.1527042
7: -0.5173369, 0.2315183, -0.5173369, 0.2315183, -0.1313275, 0.1312985
8: -3.5481367, -1.8033185, -3.5481367, -1.8033185, -0.6313135, 0.6320578
9: -4.2394500, -2.5951118, -4.2394500, -2.5951118, -0.5918715, 0.5927645

Time for backsubstitution: 6.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 516
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 431
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 856
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3220
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 3531

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2391

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0293273, upper bound: 0.0293865
time: 11.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0293815, upper bound: 0.0293272
time: 111.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.0459356, -1.4632723, -3.0459356, -1.4632723, -0.5163734, 0.5167283
1: -2.2329955, -0.5492516, -2.2329955, -0.5492516, -0.5895088, 0.5899131
2: -2.5440857, -1.8995905, -2.5440857, -1.8995905, -0.1443043, 0.1443114
3: -0.8581853, -0.5421180, -0.8581853, -0.5421180, -0.0766307, 0.0766245
4: -2.4013999, -1.4624283, -2.4013999, -1.4624283, -0.2028663, 0.2028738
5: -0.4060171, -0.1588888, -0.4060171, -0.1588888, -0.0566835, 0.0566692
6: -1.2678086, -0.6738276, -1.2678086, -0.6738276, -0.1527140, 0.1527045
7: -0.5173369, 0.2315183, -0.5173369, 0.2315183, -0.1312938, 0.1313135
8: -3.5481367, -1.8033185, -3.5481367, -1.8033185, -0.6314054, 0.6316900
9: -4.2394500, -2.5951118, -4.2394500, -2.5951118, -0.5919678, 0.5923283

Time for backsubstitution: 6.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 516
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 431
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 856
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3220
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 3531

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2391

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0293273, upper bound: 0.0293873
time: 12.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0293815, upper bound: 0.0293272
time: 117.90 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 137.09 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 137.09
Output dim: 3, lower bound: -0.0293188, upper bound: 0.0293846
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 137.09
Output dim: 3, lower bound: -0.0293731, upper bound: 0.0293428
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 137.09
Output dim: 3, lower bound: -0.0293188, upper bound: 0.0293847
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 137.09
Output dim: 3, lower bound: -0.0293731, upper bound: 0.0293428
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 137.09
Output dim: 3, lower bound: -0.0293273, upper bound: 0.0293865
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 137.09
Output dim: 3, lower bound: -0.0293815, upper bound: 0.0293272
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 137.09
Output dim: 3, lower bound: -0.0293273, upper bound: 0.0293873
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 137.09
Output dim: 3, lower bound: -0.0293815, upper bound: 0.0293272

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.0459356, -1.4632723, -3.0459356, -1.4632723, -0.5057863, 0.5050299
1: -2.2329955, -0.5492516, -2.2329955, -0.5492516, -0.5807899, 0.5804155
2: -2.5440857, -1.8995905, -2.5440857, -1.8995905, -0.1435578, 0.1435372
3: -0.8581853, -0.5421180, -0.8581853, -0.5421180, -0.0760577, 0.0761000
4: -2.4013999, -1.4624283, -2.4013999, -1.4624283, -0.1981396, 0.1983136
5: -0.4060171, -0.1588888, -0.4060171, -0.1588888, -0.0560835, 0.0561186
6: -1.2678086, -0.6738276, -1.2678086, -0.6738276, -0.1498335, 0.1500093
7: -0.5173369, 0.2315183, -0.5173369, 0.2315183, -0.1309372, 0.1308976
8: -3.5481367, -1.8033185, -3.5481367, -1.8033185, -0.6260581, 0.6258733
9: -4.2394500, -2.5951118, -4.2394500, -2.5951118, -0.5878421, 0.5877328

Time for backsubstitution: 6.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 516
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 431
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 856
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3220
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 3531

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2359

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0292652, upper bound: 0.0293743
time: 7.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0293008, upper bound: 0.0292768
time: 63.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.0459356, -1.4632723, -3.0459356, -1.4632723, -0.5053094, 0.5054620
1: -2.2329955, -0.5492516, -2.2329955, -0.5492516, -0.5805019, 0.5806779
2: -2.5440857, -1.8995905, -2.5440857, -1.8995905, -0.1435432, 0.1435518
3: -0.8581853, -0.5421180, -0.8581853, -0.5421180, -0.0760885, 0.0760671
4: -2.4013999, -1.4624283, -2.4013999, -1.4624283, -0.1983373, 0.1981095
5: -0.4060171, -0.1588888, -0.4060171, -0.1588888, -0.0560997, 0.0561004
6: -1.2678086, -0.6738276, -1.2678086, -0.6738276, -0.1500000, 0.1498428
7: -0.5173369, 0.2315183, -0.5173369, 0.2315183, -0.1309316, 0.1309032
8: -3.5481367, -1.8033185, -3.5481367, -1.8033185, -0.6258963, 0.6260124
9: -4.2394500, -2.5951118, -4.2394500, -2.5951118, -0.5877705, 0.5878043

Time for backsubstitution: 6.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 516
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 431
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 856
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3220
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 3531

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2359

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0293198, upper bound: 0.0293128
time: 124.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0293549, upper bound: 0.0292801
time: 64.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.0459356, -1.4632723, -3.0459356, -1.4632723, -0.5058244, 0.5049258
1: -2.2329955, -0.5492516, -2.2329955, -0.5492516, -0.5808753, 0.5800204
2: -2.5440857, -1.8995905, -2.5440857, -1.8995905, -0.1435537, 0.1435421
3: -0.8581853, -0.5421180, -0.8581853, -0.5421180, -0.0760579, 0.0761034
4: -2.4013999, -1.4624283, -2.4013999, -1.4624283, -0.1981082, 0.1983223
5: -0.4060171, -0.1588888, -0.4060171, -0.1588888, -0.0560827, 0.0561220
6: -1.2678086, -0.6738276, -1.2678086, -0.6738276, -0.1498330, 0.1500098
7: -0.5173369, 0.2315183, -0.5173369, 0.2315183, -0.1309079, 0.1309126
8: -3.5481367, -1.8033185, -3.5481367, -1.8033185, -0.6261544, 0.6255198
9: -4.2394500, -2.5951118, -4.2394500, -2.5951118, -0.5879480, 0.5873138

Time for backsubstitution: 6.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 516
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 431
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 856
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3220
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 3531

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 2359

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0292652, upper bound: 0.0293743
time: 7.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0293008, upper bound: 0.0292836
time: 122.11 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.0459356, -1.4632723, -3.0459356, -1.4632723, -0.5053475, 0.5054027
1: -2.2329955, -0.5492516, -2.2329955, -0.5492516, -0.5805874, 0.5803083
2: -2.5440857, -1.8995905, -2.5440857, -1.8995905, -0.1435391, 0.1435567
3: -0.8581853, -0.5421180, -0.8581853, -0.5421180, -0.0760907, 0.0760706
4: -2.4013999, -1.4624283, -2.4013999, -1.4624283, -0.1983124, 0.1981182
5: -0.4060171, -0.1588888, -0.4060171, -0.1588888, -0.0561009, 0.0561038
6: -1.2678086, -0.6738276, -1.2678086, -0.6738276, -0.1499995, 0.1498433
7: -0.5173369, 0.2315183, -0.5173369, 0.2315183, -0.1309023, 0.1309182
8: -3.5481367, -1.8033185, -3.5481367, -1.8033185, -0.6259926, 0.6256817
9: -4.2394500, -2.5951118, -4.2394500, -2.5951118, -0.5878764, 0.5873853

Time for backsubstitution: 6.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 516
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 431
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 856
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3220
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 3531

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 2359

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0293198, upper bound: 0.0293128
time: 124.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0293549, upper bound: 0.0292772
time: 203.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.0459356, -1.4632723, -3.0459356, -1.4632723, -0.5054027, 0.5054086
1: -2.2329955, -0.5492516, -2.2329955, -0.5492516, -0.5803084, 0.5809051
2: -2.5440857, -1.8995905, -2.5440857, -1.8995905, -0.1435559, 0.1435391
3: -0.8581853, -0.5421180, -0.8581853, -0.5421180, -0.0760675, 0.0760907
4: -2.4013999, -1.4624283, -2.4013999, -1.4624283, -0.1981409, 0.1983124
5: -0.4060171, -0.1588888, -0.4060171, -0.1588888, -0.0561015, 0.0561009
6: -1.2678086, -0.6738276, -1.2678086, -0.6738276, -0.1498433, 0.1499995
7: -0.5173369, 0.2315183, -0.5173369, 0.2315183, -0.1309369, 0.1309023
8: -3.5481367, -1.8033185, -3.5481367, -1.8033185, -0.6256816, 0.6262642
9: -4.2394500, -2.5951118, -4.2394500, -2.5951118, -0.5873853, 0.5882066

Time for backsubstitution: 6.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 516
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 431
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 856
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3220
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 3531

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2359

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0292744, upper bound: 0.0293047
time: 27.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0293072, upper bound: 0.0292669
time: 51.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.0459356, -1.4632723, -3.0459356, -1.4632723, -0.5049257, 0.5058408
1: -2.2329955, -0.5492516, -2.2329955, -0.5492516, -0.5800204, 0.5811675
2: -2.5440857, -1.8995905, -2.5440857, -1.8995905, -0.1435413, 0.1435537
3: -0.8581853, -0.5421180, -0.8581853, -0.5421180, -0.0760982, 0.0760579
4: -2.4013999, -1.4624283, -2.4013999, -1.4624283, -0.1983386, 0.1981082
5: -0.4060171, -0.1588888, -0.4060171, -0.1588888, -0.0561176, 0.0560827
6: -1.2678086, -0.6738276, -1.2678086, -0.6738276, -0.1500098, 0.1498330
7: -0.5173369, 0.2315183, -0.5173369, 0.2315183, -0.1309313, 0.1309079
8: -3.5481367, -1.8033185, -3.5481367, -1.8033185, -0.6255199, 0.6264034
9: -4.2394500, -2.5951118, -4.2394500, -2.5951118, -0.5873138, 0.5882783

Time for backsubstitution: 6.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 516
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 431
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 856
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3220
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 3531

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2359

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0293289, upper bound: 0.0293059
time: 71.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0293613, upper bound: 0.0292731
time: 107.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.0459356, -1.4632723, -3.0459356, -1.4632723, -0.5054408, 0.5053094
1: -2.2329955, -0.5492516, -2.2329955, -0.5492516, -0.5803939, 0.5805019
2: -2.5440857, -1.8995905, -2.5440857, -1.8995905, -0.1435518, 0.1435440
3: -0.8581853, -0.5421180, -0.8581853, -0.5421180, -0.0760671, 0.0760942
4: -2.4013999, -1.4624283, -2.4013999, -1.4624283, -0.1981095, 0.1983211
5: -0.4060171, -0.1588888, -0.4060171, -0.1588888, -0.0561004, 0.0561044
6: -1.2678086, -0.6738276, -1.2678086, -0.6738276, -0.1498428, 0.1500000
7: -0.5173369, 0.2315183, -0.5173369, 0.2315183, -0.1309032, 0.1309173
8: -3.5481367, -1.8033185, -3.5481367, -1.8033185, -0.6257779, 0.6258963
9: -4.2394500, -2.5951118, -4.2394500, -2.5951118, -0.5874912, 0.5877705

Time for backsubstitution: 6.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 516
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 431
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 856
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3220
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 3531

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2359

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0292744, upper bound: 0.0293582
time: 13.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0293072, upper bound: 0.0293208
time: 38.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.0459356, -1.4632723, -3.0459356, -1.4632723, -0.5049638, 0.5057864
1: -2.2329955, -0.5492516, -2.2329955, -0.5492516, -0.5801059, 0.5807898
2: -2.5440857, -1.8995905, -2.5440857, -1.8995905, -0.1435372, 0.1435586
3: -0.8581853, -0.5421180, -0.8581853, -0.5421180, -0.0761000, 0.0760613
4: -2.4013999, -1.4624283, -2.4013999, -1.4624283, -0.1983136, 0.1981169
5: -0.4060171, -0.1588888, -0.4060171, -0.1588888, -0.0561186, 0.0560862
6: -1.2678086, -0.6738276, -1.2678086, -0.6738276, -0.1500093, 0.1498335
7: -0.5173369, 0.2315183, -0.5173369, 0.2315183, -0.1308976, 0.1309229
8: -3.5481367, -1.8033185, -3.5481367, -1.8033185, -0.6256161, 0.6260582
9: -4.2394500, -2.5951118, -4.2394500, -2.5951118, -0.5874196, 0.5878421

Time for backsubstitution: 6.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 516
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 431
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 856
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3220
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 3531

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 2359

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0293289, upper bound: 0.0293066
time: 67.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0293613, upper bound: 0.0292724
time: 119.06 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 193.87 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 193.87
Output dim: 3, lower bound: -0.0292652, upper bound: 0.0293743
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 193.87
Output dim: 3, lower bound: -0.0293008, upper bound: 0.0292768
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 193.87
Output dim: 3, lower bound: -0.0293198, upper bound: 0.0293128
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 193.87
Output dim: 3, lower bound: -0.0293549, upper bound: 0.0292801
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 193.87
Output dim: 3, lower bound: -0.0292652, upper bound: 0.0293743
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 193.87
Output dim: 3, lower bound: -0.0293008, upper bound: 0.0292836
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 193.87
Output dim: 3, lower bound: -0.0293198, upper bound: 0.0293128
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 193.87
Output dim: 3, lower bound: -0.0293549, upper bound: 0.0292772
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 193.87
Output dim: 3, lower bound: -0.0292744, upper bound: 0.0293047
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 193.87
Output dim: 3, lower bound: -0.0293072, upper bound: 0.0292669
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 193.87
Output dim: 3, lower bound: -0.0293289, upper bound: 0.0293059
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 193.87
Output dim: 3, lower bound: -0.0293613, upper bound: 0.0292731
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 193.87
Output dim: 3, lower bound: -0.0292744, upper bound: 0.0293582
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 193.87
Output dim: 3, lower bound: -0.0293072, upper bound: 0.0293208
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 193.87
Output dim: 3, lower bound: -0.0293289, upper bound: 0.0293066
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 193.87
Output dim: 3, lower bound: -0.0293613, upper bound: 0.0292724

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.0459356, -1.4632723, -3.0459356, -1.4632723, -0.4886516, 0.4872814
1: -2.2329955, -0.5492516, -2.2329955, -0.5492516, -0.5630091, 0.5619496
2: -2.5440857, -1.8995905, -2.5440857, -1.8995905, -0.1434424, 0.1434341
3: -0.8581853, -0.5421180, -0.8581853, -0.5421180, -0.0753923, 0.0754806
4: -2.4013999, -1.4624283, -2.4013999, -1.4624283, -0.1981079, 0.1982825
5: -0.4060171, -0.1588888, -0.4060171, -0.1588888, -0.0552197, 0.0553047
6: -1.2678086, -0.6738276, -1.2678086, -0.6738276, -0.1485273, 0.1487938
7: -0.5173369, 0.2315183, -0.5173369, 0.2315183, -0.1308134, 0.1307599
8: -3.5481367, -1.8033185, -3.5481367, -1.8033185, -0.6161534, 0.6157597
9: -4.2394500, -2.5951118, -4.2394500, -2.5951118, -0.5747632, 0.5742404

Time for backsubstitution: 6.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 516
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 431
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 856
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3220
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 3531

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 125

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2601

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0292535, upper bound: 0.0293468
time: 72.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0292489, upper bound: 0.0293521
time: 92.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.0459356, -1.4632723, -3.0459356, -1.4632723, -0.4875610, 0.4883626
1: -2.2329955, -0.5492516, -2.2329955, -0.5492516, -0.5620359, 0.5629389
2: -2.5440857, -1.8995905, -2.5440857, -1.8995905, -0.1434404, 0.1434364
3: -0.8581853, -0.5421180, -0.8581853, -0.5421180, -0.0754705, 0.0754017
4: -2.4013999, -1.4624283, -2.4013999, -1.4624283, -0.1983062, 0.1980777
5: -0.4060171, -0.1588888, -0.4060171, -0.1588888, -0.0552880, 0.0552365
6: -1.2678086, -0.6738276, -1.2678086, -0.6738276, -0.1487855, 0.1485366
7: -0.5173369, 0.2315183, -0.5173369, 0.2315183, -0.1307940, 0.1307794
8: -3.5481367, -1.8033185, -3.5481367, -1.8033185, -0.6157829, 0.6161391
9: -4.2394500, -2.5951118, -4.2394500, -2.5951118, -0.5742781, 0.5747622

Time for backsubstitution: 6.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 516
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 431
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 856
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3220
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 3531

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 125

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2601

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0293437, upper bound: 0.0292607
time: 41.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0293386, upper bound: 0.0292768
time: 7.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.0459356, -1.4632723, -3.0459356, -1.4632723, -0.4886818, 0.4871773
1: -2.2329955, -0.5492516, -2.2329955, -0.5492516, -0.5630971, 0.5615544
2: -2.5440857, -1.8995905, -2.5440857, -1.8995905, -0.1434383, 0.1434391
3: -0.8581853, -0.5421180, -0.8581853, -0.5421180, -0.0753924, 0.0754835
4: -2.4013999, -1.4624283, -2.4013999, -1.4624283, -0.1980765, 0.1982935
5: -0.4060171, -0.1588888, -0.4060171, -0.1588888, -0.0552189, 0.0553089
6: -1.2678086, -0.6738276, -1.2678086, -0.6738276, -0.1485268, 0.1487941
7: -0.5173369, 0.2315183, -0.5173369, 0.2315183, -0.1307841, 0.1307754
8: -3.5481367, -1.8033185, -3.5481367, -1.8033185, -0.6162658, 0.6154064
9: -4.2394500, -2.5951118, -4.2394500, -2.5951118, -0.5748768, 0.5738212

Time for backsubstitution: 6.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 516
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 431
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 856
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3220
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 3531

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 125

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2601

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0292535, upper bound: 0.0293468
time: 71.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0292489, upper bound: 0.0293601
time: 60.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.0459356, -1.4632723, -3.0459356, -1.4632723, -0.4875911, 0.4883220
1: -2.2329955, -0.5492516, -2.2329955, -0.5492516, -0.5621238, 0.5625939
2: -2.5440857, -1.8995905, -2.5440857, -1.8995905, -0.1434375, 0.1434414
3: -0.8581853, -0.5421180, -0.8581853, -0.5421180, -0.0754741, 0.0754046
4: -2.4013999, -1.4624283, -2.4013999, -1.4624283, -0.1982815, 0.1980887
5: -0.4060171, -0.1588888, -0.4060171, -0.1588888, -0.0552901, 0.0552408
6: -1.2678086, -0.6738276, -1.2678086, -0.6738276, -0.1487856, 0.1485369
7: -0.5173369, 0.2315183, -0.5173369, 0.2315183, -0.1307677, 0.1307950
8: -3.5481367, -1.8033185, -3.5481367, -1.8033185, -0.6158952, 0.6158305
9: -4.2394500, -2.5951118, -4.2394500, -2.5951118, -0.5743917, 0.5743660

Time for backsubstitution: 6.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 516
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 431
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 856
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3220
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 3531

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 125

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2601

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0293437, upper bound: 0.0292659
time: 24.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0293386, upper bound: 0.0292768
time: 7.37 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.0459356, -1.4632723, -3.0459356, -1.4632723, -0.4878450, 0.4880924
1: -2.2329955, -0.5492516, -2.2329955, -0.5492516, -0.5623059, 0.5627014
2: -2.5440857, -1.8995905, -2.5440857, -1.8995905, -0.1434259, 0.1434521
3: -0.8581853, -0.5421180, -0.8581853, -0.5421180, -0.0754327, 0.0754413
4: -2.4013999, -1.4624283, -2.4013999, -1.4624283, -0.1983068, 0.1980774
5: -0.4060171, -0.1588888, -0.4060171, -0.1588888, -0.0552537, 0.0552719
6: -1.2678086, -0.6738276, -1.2678086, -0.6738276, -0.1487036, 0.1486191
7: -0.5173369, 0.2315183, -0.5173369, 0.2315183, -0.1308075, 0.1307733
8: -3.5481367, -1.8033185, -3.5481367, -1.8033185, -0.6156688, 0.6162899
9: -4.2394500, -2.5951118, -4.2394500, -2.5951118, -0.5742946, 0.5747859

Time for backsubstitution: 6.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 516
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 431
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 856
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3220
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 3531

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 125

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2601

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0293177, upper bound: 0.0292972
time: 8.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0293126, upper bound: 0.0292935
time: 182.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.0459356, -1.4632723, -3.0459356, -1.4632723, -0.4871773, 0.4886763
1: -2.2329955, -0.5492516, -2.2329955, -0.5492516, -0.5615544, 0.5633486
2: -2.5440857, -1.8995905, -2.5440857, -1.8995905, -0.1434370, 0.1434383
3: -0.8581853, -0.5421180, -0.8581853, -0.5421180, -0.0754774, 0.0753924
4: -2.4013999, -1.4624283, -2.4013999, -1.4624283, -0.1983075, 0.1980765
5: -0.4060171, -0.1588888, -0.4060171, -0.1588888, -0.0553022, 0.0552189
6: -1.2678086, -0.6738276, -1.2678086, -0.6738276, -0.1487938, 0.1485268
7: -0.5173369, 0.2315183, -0.5173369, 0.2315183, -0.1307890, 0.1307841
8: -3.5481367, -1.8033185, -3.5481367, -1.8033185, -0.6154064, 0.6164651
9: -4.2394500, -2.5951118, -4.2394500, -2.5951118, -0.5738213, 0.5751626

Time for backsubstitution: 6.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 516
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 431
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 856
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3220
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 3531

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 125

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2601

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0293499, upper bound: 0.0292649
time: 8.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0293451, upper bound: 0.0292571
time: 100.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.0459356, -1.4632723, -3.0459356, -1.4632723, -0.4883522, 0.4875609
1: -2.2329955, -0.5492516, -2.2329955, -0.5492516, -0.5626818, 0.5620359
2: -2.5440857, -1.8995905, -2.5440857, -1.8995905, -0.1434364, 0.1434425
3: -0.8581853, -0.5421180, -0.8581853, -0.5421180, -0.0754017, 0.0754771
4: -2.4013999, -1.4624283, -2.4013999, -1.4624283, -0.1980777, 0.1982925
5: -0.4060171, -0.1588888, -0.4060171, -0.1588888, -0.0552365, 0.0552944
6: -1.2678086, -0.6738276, -1.2678086, -0.6738276, -0.1485366, 0.1487859
7: -0.5173369, 0.2315183, -0.5173369, 0.2315183, -0.1307794, 0.1307833
8: -3.5481367, -1.8033185, -3.5481367, -1.8033185, -0.6159430, 0.6157829
9: -4.2394500, -2.5951118, -4.2394500, -2.5951118, -0.5744796, 0.5742781

Time for backsubstitution: 6.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 516
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 431
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 856
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3220
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 3531

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 125

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2601

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0292625, upper bound: 0.0293428
time: 205.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0292580, upper bound: 0.0293432
time: 19.87 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.0459356, -1.4632723, -3.0459356, -1.4632723, -0.4878752, 0.4880379
1: -2.2329955, -0.5492516, -2.2329955, -0.5492516, -0.5623938, 0.5623239
2: -2.5440857, -1.8995905, -2.5440857, -1.8995905, -0.1434218, 0.1434571
3: -0.8581853, -0.5421180, -0.8581853, -0.5421180, -0.0754345, 0.0754442
4: -2.4013999, -1.4624283, -2.4013999, -1.4624283, -0.1982819, 0.1980884
5: -0.4060171, -0.1588888, -0.4060171, -0.1588888, -0.0552547, 0.0552762
6: -1.2678086, -0.6738276, -1.2678086, -0.6738276, -0.1487031, 0.1486194
7: -0.5173369, 0.2315183, -0.5173369, 0.2315183, -0.1307738, 0.1307889
8: -3.5481367, -1.8033185, -3.5481367, -1.8033185, -0.6157812, 0.6159445
9: -4.2394500, -2.5951118, -4.2394500, -2.5951118, -0.5744081, 0.5743496

Time for backsubstitution: 6.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 516
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 431
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 856
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3220
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 3531

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 125

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2601

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0293177, upper bound: 0.0292976
time: 7.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0293126, upper bound: 0.0292526
time: 259.63 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 276.37 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 276.37
Output dim: 3, lower bound: -0.0292535, upper bound: 0.0293468
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 276.37
Output dim: 3, lower bound: -0.0292489, upper bound: 0.0293521
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 276.37
Output dim: 3, lower bound: -0.0293437, upper bound: 0.0292607
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 276.37
Output dim: 3, lower bound: -0.0293386, upper bound: 0.0292768
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 276.37
Output dim: 3, lower bound: -0.0292535, upper bound: 0.0293468
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 276.37
Output dim: 3, lower bound: -0.0292489, upper bound: 0.0293601
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 276.37
Output dim: 3, lower bound: -0.0293437, upper bound: 0.0292659
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 276.37
Output dim: 3, lower bound: -0.0293386, upper bound: 0.0292768
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 276.37
Output dim: 3, lower bound: -0.0293177, upper bound: 0.0292972
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 276.37
Output dim: 3, lower bound: -0.0293126, upper bound: 0.0292935
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 276.37
Output dim: 3, lower bound: -0.0293499, upper bound: 0.0292649
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 276.37
Output dim: 3, lower bound: -0.0293451, upper bound: 0.0292571
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 276.37
Output dim: 3, lower bound: -0.0292625, upper bound: 0.0293428
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 276.37
Output dim: 3, lower bound: -0.0292580, upper bound: 0.0293432
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 276.37
Output dim: 3, lower bound: -0.0293177, upper bound: 0.0292976
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 276.37
Output dim: 3, lower bound: -0.0293126, upper bound: 0.0292526
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 276.37
Output dim: 3, lower bound: -0.0293613, upper bound: 0.0292724

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 45.28 + 3739.34 = 3784.62 seconds
