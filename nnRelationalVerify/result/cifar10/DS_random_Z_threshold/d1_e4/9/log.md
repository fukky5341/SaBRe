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
execution time: IAR + RelationalAnalysis = 7.94 + 74.06 = 82.01 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0187163, upper bound: 0.0187147

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 316
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 3255
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 526
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2207

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2327

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0187150, upper bound: 0.0187141
time: 134.89 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0187155, upper bound: 0.0187136
time: 147.97 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 282.88 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 282.88
Output dim: 8, lower bound: -0.0187150, upper bound: 0.0187141
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 282.88
Output dim: 8, lower bound: -0.0187155, upper bound: 0.0187136

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.1338015, 0.4023824, -0.1338015, 0.4023824, -0.4408549, 0.4408544
1: -0.4360359, -0.0049183, -0.4360359, -0.0049183, -0.3238196, 0.3238193
2: -3.8433530, -2.8086581, -3.8433530, -2.8086581, -0.5201365, 0.5201341
3: -6.4341688, -4.9186339, -6.4341688, -4.9186339, -0.6215711, 0.6215825
4: -5.0087810, -3.5420136, -5.0087810, -3.5420136, -0.6042608, 0.6042540
5: -7.0995064, -5.5397024, -7.0995064, -5.5397024, -0.5866101, 0.5866224
6: -8.9357405, -7.3275676, -8.9357405, -7.3275676, -0.3680786, 0.3680935
7: -3.6018801, -1.8663328, -3.6018801, -1.8663328, -0.5469568, 0.5469575
8: 0.5656098, 0.9191978, 0.5656098, 0.9191978, -0.1635269, 0.1635272
9: 0.5901058, 0.9247624, 0.5901058, 0.9247624, -0.2543772, 0.2543771

Time for backsubstitution: 6.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 526
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 3255
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 316
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2439

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2080

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0187143, upper bound: 0.0187136
time: 29.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0187141, upper bound: 0.0186931
time: 10.40 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.1338015, 0.4023824, -0.1338015, 0.4023824, -0.4408544, 0.4408548
1: -0.4360359, -0.0049183, -0.4360359, -0.0049183, -0.3238193, 0.3238196
2: -3.8433530, -2.8086581, -3.8433530, -2.8086581, -0.5201342, 0.5201366
3: -6.4341688, -4.9186339, -6.4341688, -4.9186339, -0.6215825, 0.6215711
4: -5.0087810, -3.5420136, -5.0087810, -3.5420136, -0.6042540, 0.6042608
5: -7.0995064, -5.5397024, -7.0995064, -5.5397024, -0.5866224, 0.5866101
6: -8.9357405, -7.3275676, -8.9357405, -7.3275676, -0.3680934, 0.3680786
7: -3.6018801, -1.8663328, -3.6018801, -1.8663328, -0.5469575, 0.5469568
8: 0.5656098, 0.9191978, 0.5656098, 0.9191978, -0.1635272, 0.1635270
9: 0.5901058, 0.9247624, 0.5901058, 0.9247624, -0.2543771, 0.2543772

Time for backsubstitution: 6.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 316
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 3255
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 526
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2031

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 874

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0187118, upper bound: 0.0187087
time: 161.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0187118, upper bound: 0.0187072
time: 142.03 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 309.49 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 309.49
Output dim: 8, lower bound: -0.0187143, upper bound: 0.0187136
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 309.49
Output dim: 8, lower bound: -0.0187141, upper bound: 0.0186931
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 309.49
Output dim: 8, lower bound: -0.0187118, upper bound: 0.0187087
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 309.49
Output dim: 8, lower bound: -0.0187118, upper bound: 0.0187072

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1338015, 0.4023824, -0.1338015, 0.4023824, -0.4408119, 0.4408081
1: -0.4360359, -0.0049183, -0.4360359, -0.0049183, -0.3238176, 0.3238173
2: -3.8433530, -2.8086581, -3.8433530, -2.8086581, -0.5196012, 0.5195994
3: -6.4341688, -4.9186339, -6.4341688, -4.9186339, -0.6213555, 0.6213729
4: -5.0087810, -3.5420136, -5.0087810, -3.5420136, -0.6041448, 0.6041440
5: -7.0995064, -5.5397024, -7.0995064, -5.5397024, -0.5865205, 0.5865364
6: -8.9357405, -7.3275676, -8.9357405, -7.3275676, -0.3667252, 0.3667664
7: -3.6018801, -1.8663328, -3.6018801, -1.8663328, -0.5458990, 0.5458742
8: 0.5656098, 0.9191978, 0.5656098, 0.9191978, -0.1634875, 0.1634913
9: 0.5901058, 0.9247624, 0.5901058, 0.9247624, -0.2543721, 0.2543722

Time for backsubstitution: 6.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 526
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 316
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 3255
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2136

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2117

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0187099, upper bound: 0.0187089
time: 112.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0187099, upper bound: 0.0187059
time: 285.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1338015, 0.4023824, -0.1338015, 0.4023824, -0.4408085, 0.4408113
1: -0.4360359, -0.0049183, -0.4360359, -0.0049183, -0.3238176, 0.3238173
2: -3.8433530, -2.8086581, -3.8433530, -2.8086581, -0.5196019, 0.5195987
3: -6.4341688, -4.9186339, -6.4341688, -4.9186339, -0.6213616, 0.6213669
4: -5.0087810, -3.5420136, -5.0087810, -3.5420136, -0.6041508, 0.6041381
5: -7.0995064, -5.5397024, -7.0995064, -5.5397024, -0.5865241, 0.5865327
6: -8.9357405, -7.3275676, -8.9357405, -7.3275676, -0.3667516, 0.3667400
7: -3.6018801, -1.8663328, -3.6018801, -1.8663328, -0.5458735, 0.5458997
8: 0.5656098, 0.9191978, 0.5656098, 0.9191978, -0.1634910, 0.1634878
9: 0.5901058, 0.9247624, 0.5901058, 0.9247624, -0.2543723, 0.2543720

Time for backsubstitution: 6.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 526
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 3255
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 316
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 821

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2272

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0187133, upper bound: 0.0187116
time: 72.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0187131, upper bound: 0.0187110
time: 147.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1338015, 0.4023824, -0.1338015, 0.4023824, -0.4408538, 0.4408553
1: -0.4360359, -0.0049183, -0.4360359, -0.0049183, -0.3238212, 0.3238192
2: -3.8433530, -2.8086581, -3.8433530, -2.8086581, -0.5201319, 0.5201429
3: -6.4341688, -4.9186339, -6.4341688, -4.9186339, -0.6215814, 0.6215777
4: -5.0087810, -3.5420136, -5.0087810, -3.5420136, -0.6042511, 0.6042699
5: -7.0995064, -5.5397024, -7.0995064, -5.5397024, -0.5866212, 0.5866164
6: -8.9357405, -7.3275676, -8.9357405, -7.3275676, -0.3680934, 0.3680800
7: -3.6018801, -1.8663328, -3.6018801, -1.8663328, -0.5469553, 0.5469621
8: 0.5656098, 0.9191978, 0.5656098, 0.9191978, -0.1635269, 0.1635265
9: 0.5901058, 0.9247624, 0.5901058, 0.9247624, -0.2543769, 0.2543771

Time for backsubstitution: 6.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 3513
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 2635
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 389
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 316
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 3227
type: DSZ, layer: 1, pos: 101
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2364
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 3255
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 526
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 2096
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2228

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0187017, upper bound: 0.0186983
time: 485.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0187017, upper bound: 0.0186971
time: 178.13 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 670.00 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 670.00
Output dim: 8, lower bound: -0.0187099, upper bound: 0.0187089
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 670.00
Output dim: 8, lower bound: -0.0187099, upper bound: 0.0187059
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 670.00
Output dim: 8, lower bound: -0.0187133, upper bound: 0.0187116
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 670.00
Output dim: 8, lower bound: -0.0187131, upper bound: 0.0187110
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 670.00
Output dim: 8, lower bound: -0.0187017, upper bound: 0.0186983
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 670.00
Output dim: 8, lower bound: -0.0187017, upper bound: 0.0186971
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 670.00
Output dim: 8, lower bound: -0.0187118, upper bound: 0.0187072

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 82.01 + 1939.53 = 2021.54 seconds
