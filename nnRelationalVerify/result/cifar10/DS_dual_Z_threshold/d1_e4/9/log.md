## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 9)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.018697583700000003


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.1338015, 0.4023824, -0.1338015, 0.4023824, -0.4408671, 0.4408672)
1: (-0.4360359, -0.0049183, -0.4360359, -0.0049183, -0.3238332, 0.3238332)
2: (-3.8433530, -2.8086581, -3.8433530, -2.8086581, -0.5201718, 0.5201719)
3: (-6.4341688, -4.9186339, -6.4341688, -4.9186339, -0.6218922, 0.6218922)
4: (-5.0087810, -3.5420136, -5.0087810, -3.5420136, -0.6043280, 0.6043280)
5: (-7.0995064, -5.5397024, -7.0995064, -5.5397024, -0.5869626, 0.5869627)
6: (-8.9357405, -7.3275676, -8.9357405, -7.3275676, -0.3683923, 0.3683923)
7: (-3.6018801, -1.8663328, -3.6018801, -1.8663328, -0.5472653, 0.5472653)
8: (0.5656098, 0.9191978, 0.5656098, 0.9191978, -0.1635407, 0.1635407)
9: (0.5901058, 0.9247624, 0.5901058, 0.9247624, -0.2543804, 0.2543805)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.77 + 73.20 = 80.97 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0187163, upper bound: 0.0187147

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 316
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 526
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 3255
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 3513

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2136

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0187139, upper bound: 0.0187132
time: 23.48 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0187129, upper bound: 0.0187083
time: 501.18 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 524.74 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 524.74
Output dim: 8, lower bound: -0.0187139, upper bound: 0.0187132
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 524.74
Output dim: 8, lower bound: -0.0187129, upper bound: 0.0187083

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.1338015, 0.4023824, -0.1338015, 0.4023824, -0.4408492, 0.4408491
1: -0.4360359, -0.0049183, -0.4360359, -0.0049183, -0.3236089, 0.3235954
2: -3.8433530, -2.8086581, -3.8433530, -2.8086581, -0.5191562, 0.5192100
3: -6.4341688, -4.9186339, -6.4341688, -4.9186339, -0.6176246, 0.6178582
4: -5.0087810, -3.5420136, -5.0087810, -3.5420136, -0.6016825, 0.6018255
5: -7.0995064, -5.5397024, -7.0995064, -5.5397024, -0.5824163, 0.5826631
6: -8.9357405, -7.3275676, -8.9357405, -7.3275676, -0.3628710, 0.3631690
7: -3.6018801, -1.8663328, -3.6018801, -1.8663328, -0.5442405, 0.5444053
8: 0.5656098, 0.9191978, 0.5656098, 0.9191978, -0.1635397, 0.1635397
9: 0.5901058, 0.9247624, 0.5901058, 0.9247624, -0.2543648, 0.2543647

Time for backsubstitution: 5.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 316
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 526
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 3255
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 3513

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 97

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0187089, upper bound: 0.0187148
time: 9.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0187141, upper bound: 0.0187074
time: 108.73 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.1338015, 0.4023824, -0.1338015, 0.4023824, -0.4408492, 0.4408492
1: -0.4360359, -0.0049183, -0.4360359, -0.0049183, -0.3235954, 0.3236089
2: -3.8433530, -2.8086581, -3.8433530, -2.8086581, -0.5192100, 0.5191562
3: -6.4341688, -4.9186339, -6.4341688, -4.9186339, -0.6178582, 0.6176248
4: -5.0087810, -3.5420136, -5.0087810, -3.5420136, -0.6018256, 0.6016825
5: -7.0995064, -5.5397024, -7.0995064, -5.5397024, -0.5826631, 0.5824163
6: -8.9357405, -7.3275676, -8.9357405, -7.3275676, -0.3631689, 0.3628711
7: -3.6018801, -1.8663328, -3.6018801, -1.8663328, -0.5444052, 0.5442405
8: 0.5656098, 0.9191978, 0.5656098, 0.9191978, -0.1635397, 0.1635397
9: 0.5901058, 0.9247624, 0.5901058, 0.9247624, -0.2543647, 0.2543647

Time for backsubstitution: 5.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 316
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 526
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 3255
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 3513

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 97

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0187089, upper bound: 0.0187121
time: 14.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0187141, upper bound: 0.0187081
time: 56.72 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 77.43 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 77.43
Output dim: 8, lower bound: -0.0187089, upper bound: 0.0187148
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 77.43
Output dim: 8, lower bound: -0.0187141, upper bound: 0.0187074
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 77.43
Output dim: 8, lower bound: -0.0187089, upper bound: 0.0187121
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 77.43
Output dim: 8, lower bound: -0.0187141, upper bound: 0.0187081

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1338015, 0.4023824, -0.1338015, 0.4023824, -0.4408374, 0.4408368
1: -0.4360359, -0.0049183, -0.4360359, -0.0049183, -0.3231703, 0.3231469
2: -3.8433530, -2.8086581, -3.8433530, -2.8086581, -0.5178025, 0.5178896
3: -6.4341688, -4.9186339, -6.4341688, -4.9186339, -0.6114587, 0.6118461
4: -5.0087810, -3.5420136, -5.0087810, -3.5420136, -0.5989983, 0.5992202
5: -7.0995064, -5.5397024, -7.0995064, -5.5397024, -0.5759562, 0.5763644
6: -8.9357405, -7.3275676, -8.9357405, -7.3275676, -0.3538419, 0.3543634
7: -3.6018801, -1.8663328, -3.6018801, -1.8663328, -0.5440506, 0.5442063
8: 0.5656098, 0.9191978, 0.5656098, 0.9191978, -0.1635356, 0.1635356
9: 0.5901058, 0.9247624, 0.5901058, 0.9247624, -0.2543754, 0.2543754

Time for backsubstitution: 5.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 316
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 526
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 3255
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 3513

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2135

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0187086, upper bound: 0.0187123
time: 74.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0187072, upper bound: 0.0187086
time: 134.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1338015, 0.4023824, -0.1338015, 0.4023824, -0.4408369, 0.4408371
1: -0.4360359, -0.0049183, -0.4360359, -0.0049183, -0.3231604, 0.3231568
2: -3.8433530, -2.8086581, -3.8433530, -2.8086581, -0.5178359, 0.5178562
3: -6.4341688, -4.9186339, -6.4341688, -4.9186339, -0.6116126, 0.6116921
4: -5.0087810, -3.5420136, -5.0087810, -3.5420136, -0.5990772, 0.5991413
5: -7.0995064, -5.5397024, -7.0995064, -5.5397024, -0.5761172, 0.5762030
6: -8.9357405, -7.3275676, -8.9357405, -7.3275676, -0.3540655, 0.3541397
7: -3.6018801, -1.8663328, -3.6018801, -1.8663328, -0.5440416, 0.5442154
8: 0.5656098, 0.9191978, 0.5656098, 0.9191978, -0.1635356, 0.1635356
9: 0.5901058, 0.9247624, 0.5901058, 0.9247624, -0.2543755, 0.2543753

Time for backsubstitution: 5.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 316
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 526
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 3255
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 3513

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2135

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0187132, upper bound: 0.0187082
time: 102.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0187128, upper bound: 0.0187083
time: 13.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1338015, 0.4023824, -0.1338015, 0.4023824, -0.4408371, 0.4408368
1: -0.4360359, -0.0049183, -0.4360359, -0.0049183, -0.3231568, 0.3231604
2: -3.8433530, -2.8086581, -3.8433530, -2.8086581, -0.5178561, 0.5178358
3: -6.4341688, -4.9186339, -6.4341688, -4.9186339, -0.6116922, 0.6116125
4: -5.0087810, -3.5420136, -5.0087810, -3.5420136, -0.5991414, 0.5990771
5: -7.0995064, -5.5397024, -7.0995064, -5.5397024, -0.5762030, 0.5761173
6: -8.9357405, -7.3275676, -8.9357405, -7.3275676, -0.3541397, 0.3540656
7: -3.6018801, -1.8663328, -3.6018801, -1.8663328, -0.5442154, 0.5440416
8: 0.5656098, 0.9191978, 0.5656098, 0.9191978, -0.1635356, 0.1635356
9: 0.5901058, 0.9247624, 0.5901058, 0.9247624, -0.2543753, 0.2543755

Time for backsubstitution: 5.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 316
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 526
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 3255
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 3513

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2135

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0187086, upper bound: 0.0187080
time: 83.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0187063, upper bound: 0.0187104
time: 209.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1338015, 0.4023824, -0.1338015, 0.4023824, -0.4408368, 0.4408374
1: -0.4360359, -0.0049183, -0.4360359, -0.0049183, -0.3231469, 0.3231703
2: -3.8433530, -2.8086581, -3.8433530, -2.8086581, -0.5178896, 0.5178024
3: -6.4341688, -4.9186339, -6.4341688, -4.9186339, -0.6118461, 0.6114587
4: -5.0087810, -3.5420136, -5.0087810, -3.5420136, -0.5992203, 0.5989982
5: -7.0995064, -5.5397024, -7.0995064, -5.5397024, -0.5763643, 0.5759562
6: -8.9357405, -7.3275676, -8.9357405, -7.3275676, -0.3543634, 0.3538418
7: -3.6018801, -1.8663328, -3.6018801, -1.8663328, -0.5442063, 0.5440506
8: 0.5656098, 0.9191978, 0.5656098, 0.9191978, -0.1635356, 0.1635356
9: 0.5901058, 0.9247624, 0.5901058, 0.9247624, -0.2543754, 0.2543754

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 316
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 526
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 3255
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 3513

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2135

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0187132, upper bound: 0.0187031
time: 244.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0187128, upper bound: 0.0187095
time: 22.57 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 273.51 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 273.51
Output dim: 8, lower bound: -0.0187086, upper bound: 0.0187123
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 273.51
Output dim: 8, lower bound: -0.0187072, upper bound: 0.0187086
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 273.51
Output dim: 8, lower bound: -0.0187132, upper bound: 0.0187082
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 273.51
Output dim: 8, lower bound: -0.0187128, upper bound: 0.0187083
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 273.51
Output dim: 8, lower bound: -0.0187086, upper bound: 0.0187080
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 273.51
Output dim: 8, lower bound: -0.0187063, upper bound: 0.0187104
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 273.51
Output dim: 8, lower bound: -0.0187132, upper bound: 0.0187031
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 273.51
Output dim: 8, lower bound: -0.0187128, upper bound: 0.0187095

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1338015, 0.4023824, -0.1338015, 0.4023824, -0.4408287, 0.4408281
1: -0.4360359, -0.0049183, -0.4360359, -0.0049183, -0.3229601, 0.3229330
2: -3.8433530, -2.8086581, -3.8433530, -2.8086581, -0.5162934, 0.5164001
3: -6.4341688, -4.9186339, -6.4341688, -4.9186339, -0.6021469, 0.6026537
4: -5.0087810, -3.5420136, -5.0087810, -3.5420136, -0.5958759, 0.5961388
5: -7.0995064, -5.5397024, -7.0995064, -5.5397024, -0.5660867, 0.5666207
6: -8.9357405, -7.3275676, -8.9357405, -7.3275676, -0.3479412, 0.3485379
7: -3.6018801, -1.8663328, -3.6018801, -1.8663328, -0.5356560, 0.5359122
8: 0.5656098, 0.9191978, 0.5656098, 0.9191978, -0.1635165, 0.1635163
9: 0.5901058, 0.9247624, 0.5901058, 0.9247624, -0.2543671, 0.2543670

Time for backsubstitution: 6.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 316
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 526
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 3255
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 3513

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 95

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0186939, upper bound: 0.0187068
time: 301.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0187065, upper bound: 0.0186995
time: 23.48 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 331.12 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 331.12
Output dim: 8, lower bound: -0.0186939, upper bound: 0.0187068
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 331.12
Output dim: 8, lower bound: -0.0187065, upper bound: 0.0186995
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 331.12
Output dim: 8, lower bound: -0.0187072, upper bound: 0.0187086
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 331.12
Output dim: 8, lower bound: -0.0187132, upper bound: 0.0187082
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 331.12
Output dim: 8, lower bound: -0.0187128, upper bound: 0.0187083
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 331.12
Output dim: 8, lower bound: -0.0187086, upper bound: 0.0187080
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 331.12
Output dim: 8, lower bound: -0.0187063, upper bound: 0.0187104
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 331.12
Output dim: 8, lower bound: -0.0187132, upper bound: 0.0187031
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 331.12
Output dim: 8, lower bound: -0.0187128, upper bound: 0.0187095

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 80.97 + 1966.78 = 2047.76 seconds
