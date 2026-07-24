## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 7)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.10315783889999999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3.5914955, -2.2146628, -3.5914955, -2.2146628, -0.5696697, 0.5696697)
1: (-4.1967015, -2.1738605, -4.1967015, -2.1738605, -0.8479316, 0.8479316)
2: (-0.7871203, -0.1651437, -0.7871203, -0.1651437, -0.5425514, 0.5425515)
3: (0.1599226, 0.6740252, 0.1599226, 0.6740252, -0.4912806, 0.4912806)
4: (-1.7076153, -0.7517804, -1.7076153, -0.7517804, -0.6395983, 0.6395983)
5: (-0.0337971, 0.5903980, -0.0337971, 0.5903980, -0.5007951, 0.5007952)
6: (-0.3379841, 0.5070201, -0.3379841, 0.5070201, -0.7854927, 0.7854928)
7: (-0.9320757, 0.4672965, -0.9320757, 0.4672965, -0.5549766, 0.5549766)
8: (-3.6381791, -1.4250402, -3.6381791, -1.4250402, -0.7067745, 0.7067746)
9: (-1.8199186, 0.0598726, -1.8199186, 0.0598726, -0.7099850, 0.7099849)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 8.25 + 174.94 = 183.19 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.1032555, upper bound: 0.1032611

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 3218
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2768
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 642
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 206
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2802
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2785
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 3507
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 3398
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2614
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 3464
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 412
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 3186
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 609
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3261
type: DSZ, layer: 1, pos: 3487
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 624
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 2636
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 2811
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 2963
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 377
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 2781
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2784
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2115

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2172

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1032315, upper bound: 0.1032604
time: 19.43 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1032490, upper bound: 0.1032391
time: 210.65 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 230.09 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 230.09
Output dim: 3, lower bound: -0.1032315, upper bound: 0.1032604
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 230.09
Output dim: 3, lower bound: -0.1032490, upper bound: 0.1032391

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -3.5914955, -2.2146628, -3.5914955, -2.2146628, -0.5677122, 0.5676482
1: -4.1967015, -2.1738605, -4.1967015, -2.1738605, -0.8454291, 0.8453134
2: -0.7871203, -0.1651437, -0.7871203, -0.1651437, -0.5425112, 0.5425084
3: 0.1599226, 0.6740252, 0.1599226, 0.6740252, -0.4912781, 0.4912782
4: -1.7076153, -0.7517804, -1.7076153, -0.7517804, -0.6395817, 0.6395823
5: -0.0337971, 0.5903980, -0.0337971, 0.5903980, -0.5007336, 0.5007344
6: -0.3379841, 0.5070201, -0.3379841, 0.5070201, -0.7854875, 0.7854866
7: -0.9320757, 0.4672965, -0.9320757, 0.4672965, -0.5547298, 0.5547407
8: -3.6381791, -1.4250402, -3.6381791, -1.4250402, -0.7038132, 0.7036868
9: -1.8199186, 0.0598726, -1.8199186, 0.0598726, -0.7075265, 0.7073997

Time for backsubstitution: 6.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2963
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 3487
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 2614
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 412
type: DSZ, layer: 1, pos: 3464
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 624
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 3398
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 2802
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2785
type: DSZ, layer: 1, pos: 3261
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2636
type: DSZ, layer: 1, pos: 2811
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 206
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 609
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 2768
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 377
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2781
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 3507
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 642
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 3186
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2784
type: DSZ, layer: 1, pos: 3218
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 2444

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1032315, upper bound: 0.1032573
time: 104.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1032315, upper bound: 0.1032575
time: 108.79 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -3.5914955, -2.2146628, -3.5914955, -2.2146628, -0.5676482, 0.5677122
1: -4.1967015, -2.1738605, -4.1967015, -2.1738605, -0.8453134, 0.8454291
2: -0.7871203, -0.1651437, -0.7871203, -0.1651437, -0.5425085, 0.5425113
3: 0.1599226, 0.6740252, 0.1599226, 0.6740252, -0.4912782, 0.4912781
4: -1.7076153, -0.7517804, -1.7076153, -0.7517804, -0.6395823, 0.6395817
5: -0.0337971, 0.5903980, -0.0337971, 0.5903980, -0.5007343, 0.5007336
6: -0.3379841, 0.5070201, -0.3379841, 0.5070201, -0.7854867, 0.7854875
7: -0.9320757, 0.4672965, -0.9320757, 0.4672965, -0.5547407, 0.5547298
8: -3.6381791, -1.4250402, -3.6381791, -1.4250402, -0.7036868, 0.7038132
9: -1.8199186, 0.0598726, -1.8199186, 0.0598726, -0.7073997, 0.7075266

Time for backsubstitution: 6.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3261
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 2784
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3507
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2614
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3487
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2636
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 3398
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 2963
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 3464
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 624
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 3218
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 609
type: DSZ, layer: 1, pos: 2811
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2781
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 642
type: DSZ, layer: 1, pos: 2785
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 206
type: DSZ, layer: 1, pos: 412
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 377
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2768
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2802
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 3186
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2400

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1032365, upper bound: 0.1032265
time: 264.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1032475, upper bound: 0.1032297
time: 144.80 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 415.75 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 415.75
Output dim: 3, lower bound: -0.1032315, upper bound: 0.1032573
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 415.75
Output dim: 3, lower bound: -0.1032315, upper bound: 0.1032575
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 415.75
Output dim: 3, lower bound: -0.1032365, upper bound: 0.1032265
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 415.75
Output dim: 3, lower bound: -0.1032475, upper bound: 0.1032297

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.5914955, -2.2146628, -3.5914955, -2.2146628, -0.5677122, 0.5676482
1: -4.1967015, -2.1738605, -4.1967015, -2.1738605, -0.8454291, 0.8453134
2: -0.7871203, -0.1651437, -0.7871203, -0.1651437, -0.5425112, 0.5425084
3: 0.1599226, 0.6740252, 0.1599226, 0.6740252, -0.4912781, 0.4912782
4: -1.7076153, -0.7517804, -1.7076153, -0.7517804, -0.6395817, 0.6395823
5: -0.0337971, 0.5903980, -0.0337971, 0.5903980, -0.5007336, 0.5007344
6: -0.3379841, 0.5070201, -0.3379841, 0.5070201, -0.7854875, 0.7854866
7: -0.9320757, 0.4672965, -0.9320757, 0.4672965, -0.5547298, 0.5547407
8: -3.6381791, -1.4250402, -3.6381791, -1.4250402, -0.7038132, 0.7036868
9: -1.8199186, 0.0598726, -1.8199186, 0.0598726, -0.7075265, 0.7073997

Time for backsubstitution: 6.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2811
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 3487
type: DSZ, layer: 1, pos: 3218
type: DSZ, layer: 1, pos: 2614
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 377
type: DSZ, layer: 1, pos: 2785
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2963
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2636
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 642
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 3507
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 3398
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 412
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 624
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3186
type: DSZ, layer: 1, pos: 2768
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 206
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2802
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 3261
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 3464
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2784
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 609
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2781
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2073

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3058

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1032248, upper bound: 0.1032561
time: 17.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1032243, upper bound: 0.1032534
time: 64.91 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.5914955, -2.2146628, -3.5914955, -2.2146628, -0.5677122, 0.5676482
1: -4.1967015, -2.1738605, -4.1967015, -2.1738605, -0.8454291, 0.8453134
2: -0.7871203, -0.1651437, -0.7871203, -0.1651437, -0.5425112, 0.5425084
3: 0.1599226, 0.6740252, 0.1599226, 0.6740252, -0.4912781, 0.4912782
4: -1.7076153, -0.7517804, -1.7076153, -0.7517804, -0.6395817, 0.6395823
5: -0.0337971, 0.5903980, -0.0337971, 0.5903980, -0.5007336, 0.5007344
6: -0.3379841, 0.5070201, -0.3379841, 0.5070201, -0.7854875, 0.7854866
7: -0.9320757, 0.4672965, -0.9320757, 0.4672965, -0.5547298, 0.5547407
8: -3.6381791, -1.4250402, -3.6381791, -1.4250402, -0.7038132, 0.7036868
9: -1.8199186, 0.0598726, -1.8199186, 0.0598726, -0.7075265, 0.7073997

Time for backsubstitution: 6.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2784
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3218
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 3398
type: DSZ, layer: 1, pos: 798
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 377
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2154
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 412
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2811
type: DSZ, layer: 1, pos: 609
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 3507
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 624
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 3186
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 3261
type: DSZ, layer: 1, pos: 2785
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 2802
type: DSZ, layer: 1, pos: 2781
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 2614
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2439
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2636
type: DSZ, layer: 1, pos: 3464
type: DSZ, layer: 1, pos: 206
type: DSZ, layer: 1, pos: 2768
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 2963
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2644
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 642
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 3487
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 2542

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2237

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1032321, upper bound: 0.1032567
time: 394.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1032321, upper bound: 0.1032583
time: 416.28 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 816.59 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 816.59
Output dim: 3, lower bound: -0.1032248, upper bound: 0.1032561
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 816.59
Output dim: 3, lower bound: -0.1032243, upper bound: 0.1032534
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 816.59
Output dim: 3, lower bound: -0.1032321, upper bound: 0.1032567
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 816.59
Output dim: 3, lower bound: -0.1032321, upper bound: 0.1032583
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 816.59
Output dim: 3, lower bound: -0.1032365, upper bound: 0.1032265
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 816.59
Output dim: 3, lower bound: -0.1032475, upper bound: 0.1032297

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 183.19 + 1770.88 = 1954.07 seconds
