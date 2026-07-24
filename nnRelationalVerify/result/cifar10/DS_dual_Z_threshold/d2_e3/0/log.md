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
execution time: IAR + RelationalAnalysis = 7.97 + 300.43 = 308.40 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0772930, upper bound: 0.0772953

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 274
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 3490
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3395
type: DSZ, layer: 1, pos: 471
type: DSZ, layer: 1, pos: 3505
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 3535
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 3536
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 3281
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 623
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 3534
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 2931
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 484
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 1053
type: DSZ, layer: 1, pos: 1083
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 1067
type: DSZ, layer: 1, pos: 1055
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 1068
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 332
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 416
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 582
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 628
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2342
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2467
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 2683
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 3089
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 3270
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 3283
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3481
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3507
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 3565

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2394

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0772840, upper bound: 0.0772455
time: 30.92 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0772439, upper bound: 0.0772444
time: 238.89 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 269.89 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 269.89
Output dim: 9, lower bound: -0.0772840, upper bound: 0.0772455
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 269.89
Output dim: 9, lower bound: -0.0772439, upper bound: 0.0772444

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -3.1054883, -0.7126517, -3.1054883, -0.7126517, -2.2144980, 2.2144947
1: -2.2428455, 1.3997645, -2.2428455, 1.3997645, -3.5246820, 3.5246801
2: -2.6620865, -1.1834685, -2.6620865, -1.1834685, -1.0037502, 1.0037495
3: -1.5533886, -0.1641763, -1.5533886, -0.1641763, -0.9303164, 0.9303890
4: -4.5136986, -2.5678062, -4.5136986, -2.5678062, -1.0143085, 1.0143256
5: -2.3069646, -0.9663664, -2.3069646, -0.9663664, -0.8894184, 0.8895028
6: -3.7558546, -1.6965876, -3.7558546, -1.6965876, -0.7497349, 0.7499043
7: -1.0840368, 0.4573994, -1.0840368, 0.4573994, -1.2123435, 1.2123435
8: -2.9001648, -1.5947698, -2.9001648, -1.5947698, -0.9147452, 0.9147457
9: 1.3056433, 1.6977270, 1.3056433, 1.6977270, -0.2897301, 0.2897300

Time for backsubstitution: 6.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 274
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 3490
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3395
type: DSZ, layer: 1, pos: 471
type: DSZ, layer: 1, pos: 3505
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 3535
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 3536
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 3281
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 623
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 3534
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 2931
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 484
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 1053
type: DSZ, layer: 1, pos: 1083
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 1067
type: DSZ, layer: 1, pos: 1055
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 1068
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 332
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 416
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 582
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 628
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2342
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2467
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 2683
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 3089
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 3270
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 3283
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3481
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3507
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 3565

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 129

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0772757, upper bound: 0.0772041
time: 290.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0772456, upper bound: 0.0772379
time: 35.96 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -3.1054883, -0.7126517, -3.1054883, -0.7126517, -2.2144949, 2.2144985
1: -2.2428455, 1.3997645, -2.2428455, 1.3997645, -3.5246801, 3.5246820
2: -2.6620865, -1.1834685, -2.6620865, -1.1834685, -1.0037495, 1.0037502
3: -1.5533886, -0.1641763, -1.5533886, -0.1641763, -0.9303891, 0.9303163
4: -4.5136986, -2.5678062, -4.5136986, -2.5678062, -1.0143256, 1.0143085
5: -2.3069646, -0.9663664, -2.3069646, -0.9663664, -0.8895029, 0.8894184
6: -3.7558546, -1.6965876, -3.7558546, -1.6965876, -0.7499043, 0.7497348
7: -1.0840368, 0.4573994, -1.0840368, 0.4573994, -1.2123435, 1.2123435
8: -2.9001648, -1.5947698, -2.9001648, -1.5947698, -0.9147456, 0.9147452
9: 1.3056433, 1.6977270, 1.3056433, 1.6977270, -0.2897300, 0.2897301

Time for backsubstitution: 6.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 274
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 3490
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3395
type: DSZ, layer: 1, pos: 471
type: DSZ, layer: 1, pos: 3505
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 3535
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 3536
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 3281
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 623
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 3534
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 2931
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 484
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 1053
type: DSZ, layer: 1, pos: 1083
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 1067
type: DSZ, layer: 1, pos: 1055
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 1068
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 332
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 416
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 582
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 628
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2342
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2467
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 2683
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 3089
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 3270
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 3283
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3481
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3507
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 3565

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 129

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0772320, upper bound: 0.0772058
time: 195.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0772056, upper bound: 0.0772775
time: 179.60 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 380.78 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 380.78
Output dim: 9, lower bound: -0.0772757, upper bound: 0.0772041
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 380.78
Output dim: 9, lower bound: -0.0772456, upper bound: 0.0772379
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 380.78
Output dim: 9, lower bound: -0.0772320, upper bound: 0.0772058
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 380.78
Output dim: 9, lower bound: -0.0772056, upper bound: 0.0772775

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.1054883, -0.7126517, -3.1054883, -0.7126517, -2.2144699, 2.2144632
1: -2.2428455, 1.3997645, -2.2428455, 1.3997645, -3.5246606, 3.5246582
2: -2.6620865, -1.1834685, -2.6620865, -1.1834685, -1.0037434, 1.0037410
3: -1.5533886, -0.1641763, -1.5533886, -0.1641763, -0.9300913, 0.9301950
4: -4.5136986, -2.5678062, -4.5136986, -2.5678062, -1.0139254, 1.0139954
5: -2.3069646, -0.9663664, -2.3069646, -0.9663664, -0.8891766, 0.8892947
6: -3.7558546, -1.6965876, -3.7558546, -1.6965876, -0.7490928, 0.7493491
7: -1.0840368, 0.4573994, -1.0840368, 0.4573994, -1.2123384, 1.2123390
8: -2.9001648, -1.5947698, -2.9001648, -1.5947698, -0.9147381, 0.9147419
9: 1.3056433, 1.6977270, 1.3056433, 1.6977270, -0.2897251, 0.2897241

Time for backsubstitution: 6.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 274
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 3490
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3395
type: DSZ, layer: 1, pos: 471
type: DSZ, layer: 1, pos: 3505
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 3535
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 3536
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 3281
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 623
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 3534
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 2931
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 484
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 1053
type: DSZ, layer: 1, pos: 1083
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 1067
type: DSZ, layer: 1, pos: 1055
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 1068
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 332
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 416
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 582
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 628
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2342
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2467
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 2683
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 3089
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 3270
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 3283
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3481
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3507
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 3565

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2139

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0772671, upper bound: 0.0771602
time: 404.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0772254, upper bound: 0.0771996
time: 81.11 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.1054883, -0.7126517, -3.1054883, -0.7126517, -2.2144666, 2.2144661
1: -2.2428455, 1.3997645, -2.2428455, 1.3997645, -3.5246606, 3.5246601
2: -2.6620865, -1.1834685, -2.6620865, -1.1834685, -1.0037417, 1.0037427
3: -1.5533886, -0.1641763, -1.5533886, -0.1641763, -0.9301224, 0.9301640
4: -4.5136986, -2.5678062, -4.5136986, -2.5678062, -1.0139784, 1.0139426
5: -2.3069646, -0.9663664, -2.3069646, -0.9663664, -0.8892105, 0.8892610
6: -3.7558546, -1.6965876, -3.7558546, -1.6965876, -0.7491801, 0.7492621
7: -1.0840368, 0.4573994, -1.0840368, 0.4573994, -1.2123392, 1.2123386
8: -2.9001648, -1.5947698, -2.9001648, -1.5947698, -0.9147413, 0.9147385
9: 1.3056433, 1.6977270, 1.3056433, 1.6977270, -0.2897242, 0.2897250

Time for backsubstitution: 6.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 274
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 3490
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3395
type: DSZ, layer: 1, pos: 471
type: DSZ, layer: 1, pos: 3505
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 3535
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 3536
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 3281
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 623
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 3534
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 2931
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 484
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 1053
type: DSZ, layer: 1, pos: 1083
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 1067
type: DSZ, layer: 1, pos: 1055
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 1068
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 332
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 416
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 582
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 628
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2342
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2467
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 2683
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 3089
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 3270
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 3283
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3481
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3507
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 3565

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 2139

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0772388, upper bound: 0.0771915
time: 22.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0772022, upper bound: 0.0772310
time: 22.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.1054883, -0.7126517, -3.1054883, -0.7126517, -2.2144666, 2.2144661
1: -2.2428455, 1.3997645, -2.2428455, 1.3997645, -3.5246592, 3.5246601
2: -2.6620865, -1.1834685, -2.6620865, -1.1834685, -1.0037427, 1.0037417
3: -1.5533886, -0.1641763, -1.5533886, -0.1641763, -0.9301640, 0.9301224
4: -4.5136986, -2.5678062, -4.5136986, -2.5678062, -1.0139426, 1.0139784
5: -2.3069646, -0.9663664, -2.3069646, -0.9663664, -0.8892610, 0.8892105
6: -3.7558546, -1.6965876, -3.7558546, -1.6965876, -0.7492623, 0.7491801
7: -1.0840368, 0.4573994, -1.0840368, 0.4573994, -1.2123387, 1.2123390
8: -2.9001648, -1.5947698, -2.9001648, -1.5947698, -0.9147385, 0.9147414
9: 1.3056433, 1.6977270, 1.3056433, 1.6977270, -0.2897250, 0.2897242

Time for backsubstitution: 6.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 274
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 3490
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3395
type: DSZ, layer: 1, pos: 471
type: DSZ, layer: 1, pos: 3505
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 3535
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 3536
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 3281
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 623
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 3534
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 2931
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 484
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 1053
type: DSZ, layer: 1, pos: 1083
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 1067
type: DSZ, layer: 1, pos: 1055
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 1068
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 332
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 416
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 582
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 628
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2342
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2467
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 2683
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 3089
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 3270
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 3283
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3481
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3507
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 3565

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 2139

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0772243, upper bound: 0.0772045
time: 96.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0771869, upper bound: 0.0772436
time: 228.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.1054883, -0.7126517, -3.1054883, -0.7126517, -2.2144637, 2.2144699
1: -2.2428455, 1.3997645, -2.2428455, 1.3997645, -3.5246587, 3.5246601
2: -2.6620865, -1.1834685, -2.6620865, -1.1834685, -1.0037410, 1.0037434
3: -1.5533886, -0.1641763, -1.5533886, -0.1641763, -0.9301950, 0.9300913
4: -4.5136986, -2.5678062, -4.5136986, -2.5678062, -1.0139954, 1.0139254
5: -2.3069646, -0.9663664, -2.3069646, -0.9663664, -0.8892947, 0.8891766
6: -3.7558546, -1.6965876, -3.7558546, -1.6965876, -0.7493492, 0.7490928
7: -1.0840368, 0.4573994, -1.0840368, 0.4573994, -1.2123392, 1.2123386
8: -2.9001648, -1.5947698, -2.9001648, -1.5947698, -0.9147417, 0.9147381
9: 1.3056433, 1.6977270, 1.3056433, 1.6977270, -0.2897241, 0.2897251

Time for backsubstitution: 6.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 274
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 3490
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3395
type: DSZ, layer: 1, pos: 471
type: DSZ, layer: 1, pos: 3505
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 3535
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 3536
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 3281
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 623
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 3534
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 2931
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 484
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 1053
type: DSZ, layer: 1, pos: 1083
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 1067
type: DSZ, layer: 1, pos: 1055
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 1068
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 332
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 416
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 582
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 628
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2342
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2467
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 2683
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 3089
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 3270
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 3283
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3481
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3507
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 3565

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2139

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0771973, upper bound: 0.0772336
time: 160.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0771584, upper bound: 0.0772077
time: 311.71 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 478.98 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 478.98
Output dim: 9, lower bound: -0.0772671, upper bound: 0.0771602
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 478.98
Output dim: 9, lower bound: -0.0772254, upper bound: 0.0771996
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 478.98
Output dim: 9, lower bound: -0.0772388, upper bound: 0.0771915
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 478.98
Output dim: 9, lower bound: -0.0772022, upper bound: 0.0772310
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 478.98
Output dim: 9, lower bound: -0.0772243, upper bound: 0.0772045
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 478.98
Output dim: 9, lower bound: -0.0771869, upper bound: 0.0772436
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 478.98
Output dim: 9, lower bound: -0.0771973, upper bound: 0.0772336
DS_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 478.98
Output dim: 9, lower bound: -0.0771584, upper bound: 0.0772077

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.1054883, -0.7126517, -3.1054883, -0.7126517, -2.2140830, 2.2140732
1: -2.2428455, 1.3997645, -2.2428455, 1.3997645, -3.5244770, 3.5244694
2: -2.6620865, -1.1834685, -2.6620865, -1.1834685, -1.0034502, 1.0034583
3: -1.5533886, -0.1641763, -1.5533886, -0.1641763, -0.9288511, 0.9290161
4: -4.5136986, -2.5678062, -4.5136986, -2.5678062, -1.0126364, 1.0127714
5: -2.3069646, -0.9663664, -2.3069646, -0.9663664, -0.8881969, 0.8883610
6: -3.7558546, -1.6965876, -3.7558546, -1.6965876, -0.7433270, 0.7438663
7: -1.0840368, 0.4573994, -1.0840368, 0.4573994, -1.2122785, 1.2122821
8: -2.9001648, -1.5947698, -2.9001648, -1.5947698, -0.9146771, 0.9146876
9: 1.3056433, 1.6977270, 1.3056433, 1.6977270, -0.2897226, 0.2897215

Time for backsubstitution: 6.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 274
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 3490
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3395
type: DSZ, layer: 1, pos: 471
type: DSZ, layer: 1, pos: 3505
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 3535
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 3536
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 3281
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 623
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 3534
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 2931
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 484
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 1053
type: DSZ, layer: 1, pos: 1083
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 1067
type: DSZ, layer: 1, pos: 1055
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 1068
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 332
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 416
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 582
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 628
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2342
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2467
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 2683
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 3089
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 3270
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 3283
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3481
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3507
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 3565

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2138

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0772630, upper bound: 0.0771326
time: 200.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0772075, upper bound: 0.0771457
time: 130.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.1054883, -0.7126517, -3.1054883, -0.7126517, -2.2140796, 2.2140770
1: -2.2428455, 1.3997645, -2.2428455, 1.3997645, -3.5244718, 3.5244751
2: -2.6620865, -1.1834685, -2.6620865, -1.1834685, -1.0034606, 1.0034478
3: -1.5533886, -0.1641763, -1.5533886, -0.1641763, -0.9289122, 0.9289548
4: -4.5136986, -2.5678062, -4.5136986, -2.5678062, -1.0127012, 1.0127065
5: -2.3069646, -0.9663664, -2.3069646, -0.9663664, -0.8882427, 0.8883152
6: -3.7558546, -1.6965876, -3.7558546, -1.6965876, -0.7436066, 0.7435833
7: -1.0840368, 0.4573994, -1.0840368, 0.4573994, -1.2122813, 1.2122794
8: -2.9001648, -1.5947698, -2.9001648, -1.5947698, -0.9146837, 0.9146807
9: 1.3056433, 1.6977270, 1.3056433, 1.6977270, -0.2897226, 0.2897215

Time for backsubstitution: 6.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 274
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 3490
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3395
type: DSZ, layer: 1, pos: 471
type: DSZ, layer: 1, pos: 3505
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 3535
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 3536
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 3281
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 623
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 3534
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 2931
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 484
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 1053
type: DSZ, layer: 1, pos: 1083
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 1067
type: DSZ, layer: 1, pos: 1055
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 1068
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 332
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 416
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 582
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 628
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2342
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2467
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 2683
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 3089
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 3270
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 3283
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3481
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3507
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 3565

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2138

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0772129, upper bound: 0.0771418
time: 44.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0771986, upper bound: 0.0771919
time: 190.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.1054883, -0.7126517, -3.1054883, -0.7126517, -2.2140796, 2.2140770
1: -2.2428455, 1.3997645, -2.2428455, 1.3997645, -3.5244765, 3.5244703
2: -2.6620865, -1.1834685, -2.6620865, -1.1834685, -1.0034485, 1.0034599
3: -1.5533886, -0.1641763, -1.5533886, -0.1641763, -0.9288822, 0.9289848
4: -4.5136986, -2.5678062, -4.5136986, -2.5678062, -1.0126894, 1.0127184
5: -2.3069646, -0.9663664, -2.3069646, -0.9663664, -0.8882308, 0.8883271
6: -3.7558546, -1.6965876, -3.7558546, -1.6965876, -0.7434142, 0.7437760
7: -1.0840368, 0.4573994, -1.0840368, 0.4573994, -1.2122792, 1.2122813
8: -2.9001648, -1.5947698, -2.9001648, -1.5947698, -0.9146803, 0.9146843
9: 1.3056433, 1.6977270, 1.3056433, 1.6977270, -0.2897216, 0.2897225

Time for backsubstitution: 6.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 274
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 3490
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3395
type: DSZ, layer: 1, pos: 471
type: DSZ, layer: 1, pos: 3505
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 3535
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 3536
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 3281
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 623
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 3534
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 2931
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 484
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 1053
type: DSZ, layer: 1, pos: 1083
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 1067
type: DSZ, layer: 1, pos: 1055
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 1068
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 332
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 416
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 582
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 628
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2342
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2467
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 2683
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 3089
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 3270
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 3283
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3481
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3507
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 3565

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2138

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0772319, upper bound: 0.0771609
time: 18.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0771833, upper bound: 0.0771746
time: 133.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.1054883, -0.7126517, -3.1054883, -0.7126517, -2.2140765, 2.2140799
1: -2.2428455, 1.3997645, -2.2428455, 1.3997645, -3.5244713, 3.5244751
2: -2.6620865, -1.1834685, -2.6620865, -1.1834685, -1.0034590, 1.0034494
3: -1.5533886, -0.1641763, -1.5533886, -0.1641763, -0.9289435, 0.9289239
4: -4.5136986, -2.5678062, -4.5136986, -2.5678062, -1.0127543, 1.0126536
5: -2.3069646, -0.9663664, -2.3069646, -0.9663664, -0.8882767, 0.8882813
6: -3.7558546, -1.6965876, -3.7558546, -1.6965876, -0.7436972, 0.7434963
7: -1.0840368, 0.4573994, -1.0840368, 0.4573994, -1.2122819, 1.2122785
8: -2.9001648, -1.5947698, -2.9001648, -1.5947698, -0.9146870, 0.9146776
9: 1.3056433, 1.6977270, 1.3056433, 1.6977270, -0.2897216, 0.2897225

Time for backsubstitution: 6.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 274
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 3490
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3395
type: DSZ, layer: 1, pos: 471
type: DSZ, layer: 1, pos: 3505
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 3535
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 3536
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 3281
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3280
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 623
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 3534
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 2931
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 484
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 1053
type: DSZ, layer: 1, pos: 1083
type: DSZ, layer: 1, pos: 1054
type: DSZ, layer: 1, pos: 1067
type: DSZ, layer: 1, pos: 1055
type: DSZ, layer: 1, pos: 3311
type: DSZ, layer: 1, pos: 1068
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 328
type: DSZ, layer: 1, pos: 332
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 416
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 582
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 628
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2342
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2467
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2533
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 2683
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 3089
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 3270
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 3283
type: DSZ, layer: 1, pos: 3298
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3313
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3481
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3507
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 3565

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2138

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0771847, upper bound: 0.0771718
time: 28.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0771695, upper bound: 0.0771751
time: 244.37 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 279.30 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 279.30
Output dim: 9, lower bound: -0.0772630, upper bound: 0.0771326
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 279.30
Output dim: 9, lower bound: -0.0772075, upper bound: 0.0771457
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 279.30
Output dim: 9, lower bound: -0.0772129, upper bound: 0.0771418
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 279.30
Output dim: 9, lower bound: -0.0771986, upper bound: 0.0771919
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 279.30
Output dim: 9, lower bound: -0.0772319, upper bound: 0.0771609
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 279.30
Output dim: 9, lower bound: -0.0771833, upper bound: 0.0771746
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 279.30
Output dim: 9, lower bound: -0.0771847, upper bound: 0.0771718
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 279.30
Output dim: 9, lower bound: -0.0771695, upper bound: 0.0771751
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 279.30
Output dim: 9, lower bound: -0.0772243, upper bound: 0.0772045
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 279.30
Output dim: 9, lower bound: -0.0771869, upper bound: 0.0772436
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 279.30
Output dim: 9, lower bound: -0.0771973, upper bound: 0.0772336

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 308.40 + 3351.63 = 3660.03 seconds
