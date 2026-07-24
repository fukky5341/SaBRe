## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 0)
Time budget: 3600 seconds
Split limit: 100
Threshold: 0.0772200027


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3.1054883, -0.7126517, -3.1054883, -0.7126517, -2.2147312, 2.2147312)
1: (-2.2428455, 1.3997645, -2.2428455, 1.3997645, -3.5248928, 3.5248928)
2: (-2.6620865, -1.1834685, -2.6620865, -1.1834685, -1.0040966, 1.0040967)
3: (-1.5533886, -0.1641763, -1.5533886, -0.1641763, -0.9330004, 0.9330003)
4: (-4.5136986, -2.5678062, -4.5136986, -2.5678062, -1.0187683, 1.0187683)
5: (-2.3069646, -0.9663664, -2.3069646, -0.9663664, -0.8923727, 0.8923725)
6: (-3.7558546, -1.6965876, -3.7558546, -1.6965876, -0.7561603, 0.7561604)
7: (-1.0840368, 0.4573994, -1.0840368, 0.4573994, -1.2123749, 1.2123750)
8: (-2.9001648, -1.5947698, -2.9001648, -1.5947698, -0.9148123, 0.9148124)
9: (1.3056433, 1.6977270, 1.3056433, 1.6977270, -0.2897561, 0.2897561)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.24 + 297.38 = 304.62 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0772930, upper bound: 0.0772953

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 274
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 3505
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 1067
type: DSZ, layer: 1, pos: 3395
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 1053
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 3281
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 623
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 332
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 1055
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2342
type: DSZ, layer: 1, pos: 471
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 1068
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 3089
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 484
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 3534
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 628
type: DSZ, layer: 1, pos: 2467
type: DSZ, layer: 1, pos: 416
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 1083
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 3507
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 3535
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 3481
type: DSZ, layer: 1, pos: 2931
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2683
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 3283
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 3490
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 582
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 3270
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 3536
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2188

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0772885, upper bound: 0.0772957
time: 20.52 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0772880, upper bound: 0.0772932
time: 23.09 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 43.63 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 43.63
Output dim: 9, lower bound: -0.0772885, upper bound: 0.0772957
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 43.63
Output dim: 9, lower bound: -0.0772880, upper bound: 0.0772932

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -3.1054883, -0.7126517, -3.1054883, -0.7126517, -2.2147267, 2.2147288
1: -2.2428455, 1.3997645, -2.2428455, 1.3997645, -3.5248365, 3.5248270
2: -2.6620865, -1.1834685, -2.6620865, -1.1834685, -1.0039158, 1.0039473
3: -1.5533886, -0.1641763, -1.5533886, -0.1641763, -0.9329120, 0.9329199
4: -4.5136986, -2.5678062, -4.5136986, -2.5678062, -1.0185797, 1.0185980
5: -2.3069646, -0.9663664, -2.3069646, -0.9663664, -0.8922527, 0.8922831
6: -3.7558546, -1.6965876, -3.7558546, -1.6965876, -0.7560704, 0.7560792
7: -1.0840368, 0.4573994, -1.0840368, 0.4573994, -1.2123008, 1.2123097
8: -2.9001648, -1.5947698, -2.9001648, -1.5947698, -0.9147741, 0.9147664
9: 1.3056433, 1.6977270, 1.3056433, 1.6977270, -0.2897539, 0.2897538

Time for backsubstitution: 5.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3283
type: DSZ, layer: 1, pos: 628
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 3505
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 1067
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 582
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 484
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3535
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 1083
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 332
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 1053
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 471
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3481
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2683
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2931
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 3281
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3270
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 274
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 416
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3089
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 1055
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 623
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2467
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3507
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 3536
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 3395
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2342
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 3534
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 3490
type: DSZ, layer: 1, pos: 1068
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 63

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3283

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0772814, upper bound: 0.0772927
time: 110.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0772881, upper bound: 0.0772831
time: 68.79 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -3.1054883, -0.7126517, -3.1054883, -0.7126517, -2.2147286, 2.2147269
1: -2.2428455, 1.3997645, -2.2428455, 1.3997645, -3.5248260, 3.5248365
2: -2.6620865, -1.1834685, -2.6620865, -1.1834685, -1.0039473, 1.0039158
3: -1.5533886, -0.1641763, -1.5533886, -0.1641763, -0.9329199, 0.9329119
4: -4.5136986, -2.5678062, -4.5136986, -2.5678062, -1.0185978, 1.0185797
5: -2.3069646, -0.9663664, -2.3069646, -0.9663664, -0.8922831, 0.8922527
6: -3.7558546, -1.6965876, -3.7558546, -1.6965876, -0.7560793, 0.7560704
7: -1.0840368, 0.4573994, -1.0840368, 0.4573994, -1.2123096, 1.2123007
8: -2.9001648, -1.5947698, -2.9001648, -1.5947698, -0.9147664, 0.9147741
9: 1.3056433, 1.6977270, 1.3056433, 1.6977270, -0.2897538, 0.2897538

Time for backsubstitution: 5.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 623
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 1068
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 1083
type: DSZ, layer: 1, pos: 2683
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 1053
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 3281
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 2467
type: DSZ, layer: 1, pos: 582
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 1067
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 3507
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 3089
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 3270
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 2931
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 3505
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 471
type: DSZ, layer: 1, pos: 332
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 3535
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 3490
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 3283
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 416
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 3536
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 3395
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 484
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 628
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 3481
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 274
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2342
type: DSZ, layer: 1, pos: 1055
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 3534
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 3017

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2027

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0772873, upper bound: 0.0772881
time: 247.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0772856, upper bound: 0.0772934
time: 292.71 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 546.20 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 546.20
Output dim: 9, lower bound: -0.0772814, upper bound: 0.0772927
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 546.20
Output dim: 9, lower bound: -0.0772881, upper bound: 0.0772831
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 546.20
Output dim: 9, lower bound: -0.0772873, upper bound: 0.0772881
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 546.20
Output dim: 9, lower bound: -0.0772856, upper bound: 0.0772934

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.1054883, -0.7126517, -3.1054883, -0.7126517, -2.2147188, 2.2147222
1: -2.2428455, 1.3997645, -2.2428455, 1.3997645, -3.5248365, 3.5248260
2: -2.6620865, -1.1834685, -2.6620865, -1.1834685, -1.0039073, 1.0039372
3: -1.5533886, -0.1641763, -1.5533886, -0.1641763, -0.9328932, 0.9328958
4: -4.5136986, -2.5678062, -4.5136986, -2.5678062, -1.0185778, 1.0185958
5: -2.3069646, -0.9663664, -2.3069646, -0.9663664, -0.8922332, 0.8922589
6: -3.7558546, -1.6965876, -3.7558546, -1.6965876, -0.7560540, 0.7560604
7: -1.0840368, 0.4573994, -1.0840368, 0.4573994, -1.2122998, 1.2123094
8: -2.9001648, -1.5947698, -2.9001648, -1.5947698, -0.9147725, 0.9147649
9: 1.3056433, 1.6977270, 1.3056433, 1.6977270, -0.2897526, 0.2897528

Time for backsubstitution: 5.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 1055
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 274
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 3490
type: DSZ, layer: 1, pos: 1053
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 3395
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2467
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3481
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 332
type: DSZ, layer: 1, pos: 2683
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 1067
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 1083
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 2342
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 3536
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 3535
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 628
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3281
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 3534
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 471
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 1068
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2931
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 3089
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 623
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 416
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 3270
type: DSZ, layer: 1, pos: 582
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 484
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 3507
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 3505
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 2289

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2585

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0772637, upper bound: 0.0772714
time: 695.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0772608, upper bound: 0.0772744
time: 46.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.1054883, -0.7126517, -3.1054883, -0.7126517, -2.2147205, 2.2147202
1: -2.2428455, 1.3997645, -2.2428455, 1.3997645, -3.5248365, 3.5248260
2: -2.6620865, -1.1834685, -2.6620865, -1.1834685, -1.0039057, 1.0039387
3: -1.5533886, -0.1641763, -1.5533886, -0.1641763, -0.9328880, 0.9329011
4: -4.5136986, -2.5678062, -4.5136986, -2.5678062, -1.0185776, 1.0185961
5: -2.3069646, -0.9663664, -2.3069646, -0.9663664, -0.8922284, 0.8922637
6: -3.7558546, -1.6965876, -3.7558546, -1.6965876, -0.7560516, 0.7560629
7: -1.0840368, 0.4573994, -1.0840368, 0.4573994, -1.2123005, 1.2123086
8: -2.9001648, -1.5947698, -2.9001648, -1.5947698, -0.9147725, 0.9147646
9: 1.3056433, 1.6977270, 1.3056433, 1.6977270, -0.2897528, 0.2897526

Time for backsubstitution: 6.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 1068
type: DSZ, layer: 1, pos: 332
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 582
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 3089
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 484
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 274
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 1055
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 3281
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 2467
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 2683
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 3270
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 471
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 416
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 3536
type: DSZ, layer: 1, pos: 2342
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 3534
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 3535
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 623
type: DSZ, layer: 1, pos: 3490
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 3395
type: DSZ, layer: 1, pos: 1067
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 3481
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 1053
type: DSZ, layer: 1, pos: 1083
type: DSZ, layer: 1, pos: 628
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 3507
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 3505
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2931
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2418

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2374

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0772782, upper bound: 0.0772309
time: 15.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0772325, upper bound: 0.0772762
time: 104.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.1054883, -0.7126517, -3.1054883, -0.7126517, -2.2143421, 2.2143407
1: -2.2428455, 1.3997645, -2.2428455, 1.3997645, -3.5246930, 3.5247073
2: -2.6620865, -1.1834685, -2.6620865, -1.1834685, -1.0039352, 1.0039058
3: -1.5533886, -0.1641763, -1.5533886, -0.1641763, -0.9320289, 0.9320490
4: -4.5136986, -2.5678062, -4.5136986, -2.5678062, -1.0180794, 1.0180444
5: -2.3069646, -0.9663664, -2.3069646, -0.9663664, -0.8914022, 0.8913997
6: -3.7558546, -1.6965876, -3.7558546, -1.6965876, -0.7542726, 0.7543198
7: -1.0840368, 0.4573994, -1.0840368, 0.4573994, -1.2119718, 1.2119515
8: -2.9001648, -1.5947698, -2.9001648, -1.5947698, -0.9144279, 0.9144359
9: 1.3056433, 1.6977270, 1.3056433, 1.6977270, -0.2897325, 0.2897325

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 1083
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 3481
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 471
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 3281
type: DSZ, layer: 1, pos: 332
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 2931
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2342
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 3507
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 3534
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 484
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2467
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 416
type: DSZ, layer: 1, pos: 3535
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 3536
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 3270
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 3490
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 623
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 1068
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 1055
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 274
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3283
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3395
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 1053
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 3089
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 628
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2683
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 3505
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 1067
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 582
type: DSZ, layer: 1, pos: 2249

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2584

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0772658, upper bound: 0.0772751
time: 149.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0772737, upper bound: 0.0772706
time: 242.95 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.1054883, -0.7126517, -3.1054883, -0.7126517, -2.2143424, 2.2143402
1: -2.2428455, 1.3997645, -2.2428455, 1.3997645, -3.5246978, 3.5247035
2: -2.6620865, -1.1834685, -2.6620865, -1.1834685, -1.0039372, 1.0039039
3: -1.5533886, -0.1641763, -1.5533886, -0.1641763, -0.9320569, 0.9320210
4: -4.5136986, -2.5678062, -4.5136986, -2.5678062, -1.0180626, 1.0180610
5: -2.3069646, -0.9663664, -2.3069646, -0.9663664, -0.8914302, 0.8913717
6: -3.7558546, -1.6965876, -3.7558546, -1.6965876, -0.7543285, 0.7542638
7: -1.0840368, 0.4573994, -1.0840368, 0.4573994, -1.2119603, 1.2119629
8: -2.9001648, -1.5947698, -2.9001648, -1.5947698, -0.9144281, 0.9144356
9: 1.3056433, 1.6977270, 1.3056433, 1.6977270, -0.2897325, 0.2897325

Time for backsubstitution: 6.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 628
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 332
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 3281
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 471
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 3490
type: DSZ, layer: 1, pos: 623
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 1053
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 3535
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3507
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 3505
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 1055
type: DSZ, layer: 1, pos: 484
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2683
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 3536
type: DSZ, layer: 1, pos: 1083
type: DSZ, layer: 1, pos: 1067
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 3481
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 3283
type: DSZ, layer: 1, pos: 274
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2342
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 416
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 3534
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2467
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 2931
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 582
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 3089
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 3395
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 1068
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 3270
type: DSZ, layer: 1, pos: 2089

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2483

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0772868, upper bound: 0.0772946
time: 16.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0772872, upper bound: 0.0772897
time: 32.41 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 55.06 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 55.06
Output dim: 9, lower bound: -0.0772637, upper bound: 0.0772714
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 55.06
Output dim: 9, lower bound: -0.0772608, upper bound: 0.0772744
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 55.06
Output dim: 9, lower bound: -0.0772782, upper bound: 0.0772309
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 55.06
Output dim: 9, lower bound: -0.0772325, upper bound: 0.0772762
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 55.06
Output dim: 9, lower bound: -0.0772658, upper bound: 0.0772751
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 55.06
Output dim: 9, lower bound: -0.0772737, upper bound: 0.0772706
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 55.06
Output dim: 9, lower bound: -0.0772868, upper bound: 0.0772946
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 55.06
Output dim: 9, lower bound: -0.0772872, upper bound: 0.0772897

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.1054883, -0.7126517, -3.1054883, -0.7126517, -2.2147012, 2.2147059
1: -2.2428455, 1.3997645, -2.2428455, 1.3997645, -3.5248270, 3.5248175
2: -2.6620865, -1.1834685, -2.6620865, -1.1834685, -1.0038692, 1.0038950
3: -1.5533886, -0.1641763, -1.5533886, -0.1641763, -0.9326218, 0.9325737
4: -4.5136986, -2.5678062, -4.5136986, -2.5678062, -1.0184553, 1.0184802
5: -2.3069646, -0.9663664, -2.3069646, -0.9663664, -0.8919449, 0.8919041
6: -3.7558546, -1.6965876, -3.7558546, -1.6965876, -0.7553664, 0.7552820
7: -1.0840368, 0.4573994, -1.0840368, 0.4573994, -1.2122908, 1.2123001
8: -2.9001648, -1.5947698, -2.9001648, -1.5947698, -0.9147671, 0.9147613
9: 1.3056433, 1.6977270, 1.3056433, 1.6977270, -0.2897443, 0.2897428

Time for backsubstitution: 5.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 332
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 628
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 484
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 3490
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 3536
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2683
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 3089
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 1067
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2342
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 3281
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 3535
type: DSZ, layer: 1, pos: 3505
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 582
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 1083
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 471
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 3270
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 274
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 623
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 3395
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2467
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 1068
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2931
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 1053
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3534
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 3507
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 416
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 3481
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 1055
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2236

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2586

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0772356, upper bound: 0.0772488
time: 18.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0772345, upper bound: 0.0772467
time: 259.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.1054883, -0.7126517, -3.1054883, -0.7126517, -2.2147021, 2.2147050
1: -2.2428455, 1.3997645, -2.2428455, 1.3997645, -3.5248270, 3.5248175
2: -2.6620865, -1.1834685, -2.6620865, -1.1834685, -1.0038650, 1.0038991
3: -1.5533886, -0.1641763, -1.5533886, -0.1641763, -0.9325709, 0.9326245
4: -4.5136986, -2.5678062, -4.5136986, -2.5678062, -1.0184622, 1.0184733
5: -2.3069646, -0.9663664, -2.3069646, -0.9663664, -0.8918785, 0.8919704
6: -3.7558546, -1.6965876, -3.7558546, -1.6965876, -0.7552756, 0.7553729
7: -1.0840368, 0.4573994, -1.0840368, 0.4573994, -1.2122905, 1.2123001
8: -2.9001648, -1.5947698, -2.9001648, -1.5947698, -0.9147691, 0.9147594
9: 1.3056433, 1.6977270, 1.3056433, 1.6977270, -0.2897427, 0.2897443

Time for backsubstitution: 5.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2342
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 484
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3507
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 332
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 3395
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 3089
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 471
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 3490
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 1055
type: DSZ, layer: 1, pos: 582
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 3536
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 3534
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 628
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 3270
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 1053
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2467
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 1083
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 274
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 3535
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 2683
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3281
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 1068
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 1067
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 623
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 2931
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 3505
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3481
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 416
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 3257

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3313

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0772597, upper bound: 0.0772730
time: 169.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0772592, upper bound: 0.0772731
time: 39.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.1054883, -0.7126517, -3.1054883, -0.7126517, -2.2145095, 2.2145085
1: -2.2428455, 1.3997645, -2.2428455, 1.3997645, -3.5247231, 3.5247102
2: -2.6620865, -1.1834685, -2.6620865, -1.1834685, -1.0033401, 1.0033765
3: -1.5533886, -0.1641763, -1.5533886, -0.1641763, -0.9287407, 0.9288213
4: -4.5136986, -2.5678062, -4.5136986, -2.5678062, -1.0169349, 1.0170274
5: -2.3069646, -0.9663664, -2.3069646, -0.9663664, -0.8869408, 0.8870561
6: -3.7558546, -1.6965876, -3.7558546, -1.6965876, -0.7506945, 0.7508033
7: -1.0840368, 0.4573994, -1.0840368, 0.4573994, -1.2122438, 1.2122533
8: -2.9001648, -1.5947698, -2.9001648, -1.5947698, -0.9148440, 0.9148378
9: 1.3056433, 1.6977270, 1.3056433, 1.6977270, -0.2897258, 0.2897253

Time for backsubstitution: 5.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 274
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 3270
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 3481
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2467
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 1055
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 3534
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2683
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 628
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 3507
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 2342
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 3536
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 1068
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 3395
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 484
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 623
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 1053
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 582
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 2931
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 471
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 1083
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 3089
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 3505
type: DSZ, layer: 1, pos: 332
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 1067
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 3490
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 3281
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 416
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 3535
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 804

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0772664, upper bound: 0.0772095
time: 16.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0772563, upper bound: 0.0772175
time: 36.89 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.1054883, -0.7126517, -3.1054883, -0.7126517, -2.2145085, 2.2145095
1: -2.2428455, 1.3997645, -2.2428455, 1.3997645, -3.5247202, 3.5247121
2: -2.6620865, -1.1834685, -2.6620865, -1.1834685, -1.0033435, 1.0033733
3: -1.5533886, -0.1641763, -1.5533886, -0.1641763, -0.9288083, 0.9287537
4: -4.5136986, -2.5678062, -4.5136986, -2.5678062, -1.0170089, 1.0169532
5: -2.3069646, -0.9663664, -2.3069646, -0.9663664, -0.8870208, 0.8869760
6: -3.7558546, -1.6965876, -3.7558546, -1.6965876, -0.7507920, 0.7507058
7: -1.0840368, 0.4573994, -1.0840368, 0.4573994, -1.2122452, 1.2122519
8: -2.9001648, -1.5947698, -2.9001648, -1.5947698, -0.9148456, 0.9148362
9: 1.3056433, 1.6977270, 1.3056433, 1.6977270, -0.2897255, 0.2897256

Time for backsubstitution: 5.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2467
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 471
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 416
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 623
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 3536
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 1055
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 3534
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3505
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 628
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 1053
type: DSZ, layer: 1, pos: 3089
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 582
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 484
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 3507
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 1067
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 3490
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3535
type: DSZ, layer: 1, pos: 332
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 3481
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 1083
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 274
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 3270
type: DSZ, layer: 1, pos: 2342
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 3395
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 1068
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2683
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2931
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 3281
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 3084

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2682

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0772319, upper bound: 0.0772761
time: 18.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0772294, upper bound: 0.0772710
time: 286.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.1054883, -0.7126517, -3.1054883, -0.7126517, -2.2143264, 2.2143307
1: -2.2428455, 1.3997645, -2.2428455, 1.3997645, -3.5246840, 3.5246978
2: -2.6620865, -1.1834685, -2.6620865, -1.1834685, -1.0038888, 1.0038724
3: -1.5533886, -0.1641763, -1.5533886, -0.1641763, -0.9318457, 0.9318122
4: -4.5136986, -2.5678062, -4.5136986, -2.5678062, -1.0180310, 1.0179951
5: -2.3069646, -0.9663664, -2.3069646, -0.9663664, -0.8911929, 0.8911477
6: -3.7558546, -1.6965876, -3.7558546, -1.6965876, -0.7539691, 0.7539011
7: -1.0840368, 0.4573994, -1.0840368, 0.4573994, -1.2119699, 1.2119501
8: -2.9001648, -1.5947698, -2.9001648, -1.5947698, -0.9144199, 0.9144276
9: 1.3056433, 1.6977270, 1.3056433, 1.6977270, -0.2897288, 0.2897283

Time for backsubstitution: 5.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 3534
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 2683
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2467
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 3395
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 484
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 3283
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 1068
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 3505
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 1055
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 1053
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 3281
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 1067
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 274
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 1083
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3536
type: DSZ, layer: 1, pos: 582
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 3535
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 332
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 623
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 3089
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 471
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 3507
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 3481
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 3270
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2931
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 3490
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 416
type: DSZ, layer: 1, pos: 628
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 2342

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3250

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0772485, upper bound: 0.0772591
time: 430.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0772627, upper bound: 0.0772595
time: 113.46 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 549.57 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 549.57
Output dim: 9, lower bound: -0.0772356, upper bound: 0.0772488
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 549.57
Output dim: 9, lower bound: -0.0772345, upper bound: 0.0772467
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 549.57
Output dim: 9, lower bound: -0.0772597, upper bound: 0.0772730
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 549.57
Output dim: 9, lower bound: -0.0772592, upper bound: 0.0772731
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 549.57
Output dim: 9, lower bound: -0.0772664, upper bound: 0.0772095
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 549.57
Output dim: 9, lower bound: -0.0772563, upper bound: 0.0772175
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 549.57
Output dim: 9, lower bound: -0.0772319, upper bound: 0.0772761
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 549.57
Output dim: 9, lower bound: -0.0772294, upper bound: 0.0772710
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 549.57
Output dim: 9, lower bound: -0.0772485, upper bound: 0.0772591
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 549.57
Output dim: 9, lower bound: -0.0772627, upper bound: 0.0772595
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 549.57
Output dim: 9, lower bound: -0.0772737, upper bound: 0.0772706
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 549.57
Output dim: 9, lower bound: -0.0772868, upper bound: 0.0772946
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 549.57
Output dim: 9, lower bound: -0.0772872, upper bound: 0.0772897

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 304.62 + 3520.56 = 3825.18 seconds
