## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 15)
Time budget: 3600 seconds
Split limit: 100
Threshold: 0.1233197568


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0302720, 1.6531119, 0.0302720, 1.6531119, -1.2186699, 1.2186699)
1: (-3.3763952, -0.8267784, -3.3763952, -0.8267784, -1.6205785, 1.6205786)
2: (-0.8847626, -0.3119067, -0.8847626, -0.3119067, -0.2177590, 0.2177590)
3: (-0.1364306, 0.2343346, -0.1364306, 0.2343346, -0.1792391, 0.1792391)
4: (-2.7811904, -2.2556832, -2.7811904, -2.2556832, -0.2461720, 0.2461720)
5: (-0.9392171, -0.5457087, -0.9392171, -0.5457087, -0.1987053, 0.1987053)
6: (-0.5227640, 0.5105959, -0.5227640, 0.5105959, -0.1530281, 0.1530281)
7: (-1.0645843, 0.0856791, -1.0645843, 0.0856791, -0.5893868, 0.5893869)
8: (-4.9083729, -3.1190684, -4.9083729, -3.1190684, -0.9098256, 0.9098256)
9: (-2.8582137, -1.3061166, -2.8582137, -1.3061166, -0.7481792, 0.7481793)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.77 + 63.55 = 71.32 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.1234413, upper bound: 0.1234429

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2570
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 262
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2391
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 3054
type: A, layer: 1, pos: 3067
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 2332
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 3052
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 3051
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 2327
type: A, layer: 1, pos: 3249
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 334
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2111
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 3003
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 3009
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 273
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2223
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 2284
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 2525
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 3430
type: A, layer: 1, pos: 2217
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 2965
type: A, layer: 1, pos: 2290
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 2055
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 3461
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 2960
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 2225
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 2233
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 2517
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 2995
type: A, layer: 1, pos: 2226
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 506
type: A, layer: 1, pos: 2480
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 3460
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 2037
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 3031
type: A, layer: 1, pos: 3030
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 2310
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2286
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 3246
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 2929
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2144
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2240
type: A, layer: 1, pos: 2241
type: A, layer: 1, pos: 2242
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2244
type: A, layer: 1, pos: 2245
type: A, layer: 1, pos: 2246
type: A, layer: 1, pos: 2247
type: A, layer: 1, pos: 2248
type: A, layer: 1, pos: 2249
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2639

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2570

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1234185, upper bound: 0.1233551
time: 7.45 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1234174, upper bound: 0.1234188
time: 146.45 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 153.96 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 153.96
Output dim: 3, lower bound: -0.1234185, upper bound: 0.1233551
NS_A2, status: Status.UNKNOWN, split count: 1, time: 153.96
Output dim: 3, lower bound: -0.1234174, upper bound: 0.1234188

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: 0.0356004, 1.6508019, 0.0347359, 1.6531117, -1.2132844, 1.2116903
1: -3.3675613, -0.8293891, -3.3688555, -0.8267794, -1.6092697, 1.6037040
2: -0.8844791, -0.3119098, -0.8845560, -0.3119071, -0.2174094, 0.2171841
3: -0.1349540, 0.2342581, -0.1363010, 0.2342727, -0.1778626, 0.1791531
4: -2.7809761, -2.2557032, -2.7811432, -2.2556987, -0.2457985, 0.2461109
5: -0.9377133, -0.5459299, -0.9391481, -0.5458881, -0.1970351, 0.1984270
6: -0.5228792, 0.5104334, -0.5227590, 0.5104630, -0.1523827, 0.1526379
7: -1.0628650, 0.0852449, -1.0645216, 0.0853255, -0.5871832, 0.5888119
8: -4.9044824, -3.1208301, -4.9051380, -3.1190684, -0.9053644, 0.9030126
9: -2.8540878, -1.3072505, -2.8546751, -1.3061171, -0.7422045, 0.7387199

Time for backsubstitution: 6.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 262
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2391
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 3067
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 2332
type: B, layer: 1, pos: 2541
type: B, layer: 1, pos: 3052
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 3051
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 2327
type: B, layer: 1, pos: 3249
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 2570
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 2585
type: B, layer: 1, pos: 145
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 334
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2111
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 3003
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 3009
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 273
type: B, layer: 1, pos: 2296
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 2223
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 2510
type: B, layer: 1, pos: 2284
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 2056
type: B, layer: 1, pos: 2666
type: B, layer: 1, pos: 2525
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 3430
type: B, layer: 1, pos: 2217
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2232
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 2965
type: B, layer: 1, pos: 2290
type: B, layer: 1, pos: 2190
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 2055
type: B, layer: 1, pos: 3461
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 2960
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 3462
type: B, layer: 1, pos: 2225
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 2353
type: B, layer: 1, pos: 2040
type: B, layer: 1, pos: 2233
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 2517
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 2995
type: B, layer: 1, pos: 2226
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2221
type: B, layer: 1, pos: 2036
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 506
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 2480
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 3460
type: B, layer: 1, pos: 2052
type: B, layer: 1, pos: 2037
type: B, layer: 1, pos: 3446
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 3031
type: B, layer: 1, pos: 3030
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 2310
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 2547
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 2286
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 3246
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 2929
type: B, layer: 1, pos: 2205
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2234
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2240
type: B, layer: 1, pos: 2241
type: B, layer: 1, pos: 2242
type: B, layer: 1, pos: 2243
type: B, layer: 1, pos: 2244
type: B, layer: 1, pos: 2245
type: B, layer: 1, pos: 2246
type: B, layer: 1, pos: 2247
type: B, layer: 1, pos: 2248
type: B, layer: 1, pos: 2249
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2639

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 2390

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233853, upper bound: 0.1232917
time: 21.21 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233852, upper bound: 0.1233206
time: 15.48 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: 0.0323961, 1.6531112, 0.0320082, 1.6531112, -1.2152257, 1.2170069
1: -3.3718610, -0.8267779, -3.3726213, -0.8267798, -1.6065719, 1.6185787
2: -0.8845055, -0.3119069, -0.8845505, -0.3119069, -0.2173949, 0.2176414
3: -0.1363644, 0.2343195, -0.1363763, 0.2343223, -0.1792089, 0.1791390
4: -2.7811644, -2.2557015, -2.7811692, -2.2556984, -0.2461445, 0.2461257
5: -0.9391818, -0.5457864, -0.9391882, -0.5457739, -0.1986723, 0.1983137
6: -0.5227616, 0.5103230, -0.5227621, 0.5103717, -0.1529928, 0.1523388
7: -1.0645516, 0.0855280, -1.0645576, 0.0855529, -0.5892990, 0.5886153
8: -4.9067578, -3.1190684, -4.9070549, -3.1190684, -0.9041972, 0.9091418
9: -2.8560424, -1.3061178, -2.8564179, -1.3061173, -0.7386556, 0.7475666

Time for backsubstitution: 6.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 262
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2391
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 3067
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 2332
type: B, layer: 1, pos: 2541
type: B, layer: 1, pos: 3052
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 3051
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 2327
type: B, layer: 1, pos: 3249
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 2570
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 2585
type: B, layer: 1, pos: 145
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 334
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2111
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 3003
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 3009
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 273
type: B, layer: 1, pos: 2296
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 2223
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 2510
type: B, layer: 1, pos: 2284
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 2056
type: B, layer: 1, pos: 2666
type: B, layer: 1, pos: 2525
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 3430
type: B, layer: 1, pos: 2217
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2232
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 2965
type: B, layer: 1, pos: 2290
type: B, layer: 1, pos: 2190
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 2055
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 3461
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 2960
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 3462
type: B, layer: 1, pos: 2225
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 2353
type: B, layer: 1, pos: 2040
type: B, layer: 1, pos: 2233
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 2517
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 2995
type: B, layer: 1, pos: 2226
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2221
type: B, layer: 1, pos: 2036
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 506
type: B, layer: 1, pos: 2480
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 3460
type: B, layer: 1, pos: 2052
type: B, layer: 1, pos: 2037
type: B, layer: 1, pos: 3446
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 3031
type: B, layer: 1, pos: 3030
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 2310
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 2547
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 2286
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 3246
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 2929
type: B, layer: 1, pos: 2205
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2234
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2240
type: B, layer: 1, pos: 2241
type: B, layer: 1, pos: 2242
type: B, layer: 1, pos: 2243
type: B, layer: 1, pos: 2244
type: B, layer: 1, pos: 2245
type: B, layer: 1, pos: 2246
type: B, layer: 1, pos: 2247
type: B, layer: 1, pos: 2248
type: B, layer: 1, pos: 2249
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2639

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2390

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233853, upper bound: 0.1233579
time: 170.35 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233847, upper bound: 0.1233875
time: 8.27 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 184.73 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 184.73
Output dim: 3, lower bound: -0.1233853, upper bound: 0.1232917
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 184.73
Output dim: 3, lower bound: -0.1233852, upper bound: 0.1233206
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 184.73
Output dim: 3, lower bound: -0.1233853, upper bound: 0.1233579
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 184.73
Output dim: 3, lower bound: -0.1233847, upper bound: 0.1233875

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: 0.0356359, 1.6477497, 0.0347791, 1.6494975, -1.2094479, 1.2085028
1: -3.3674147, -0.8354049, -3.3686805, -0.8340344, -1.6017892, 1.5975689
2: -0.8844118, -0.3122913, -0.8844746, -0.3123707, -0.2167963, 0.2166513
3: -0.1333437, 0.2342566, -0.1343579, 0.2342708, -0.1760394, 0.1769580
4: -2.7802203, -2.2557085, -2.7802222, -2.2557054, -0.2449742, 0.2451256
5: -0.9358565, -0.5459329, -0.9369200, -0.5458917, -0.1950117, 0.1959866
6: -0.5227944, 0.5104319, -0.5226531, 0.5104610, -0.1522531, 0.1524901
7: -1.0602942, 0.0852398, -1.0614096, 0.0853205, -0.5846496, 0.5857425
8: -4.9044075, -3.1240730, -4.9050493, -3.1229610, -0.9012212, 0.8996313
9: -2.8540587, -1.3112030, -2.8546400, -1.3109300, -0.7375467, 0.7349309

Time for backsubstitution: 6.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 262
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2391
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 3054
type: A, layer: 1, pos: 3067
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 2332
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 3052
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 3051
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 2327
type: A, layer: 1, pos: 3249
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 334
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2111
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 3003
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 3009
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 273
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2223
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 2284
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 3430
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 2525
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 2217
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 2965
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 2290
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 2055
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 2960
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 3461
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 2225
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 2233
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 2517
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 2226
type: A, layer: 1, pos: 2995
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 506
type: A, layer: 1, pos: 2480
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 3460
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 2037
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 3031
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 3030
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2310
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2286
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 3246
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 2929
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2144
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2240
type: A, layer: 1, pos: 2241
type: A, layer: 1, pos: 2242
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2244
type: A, layer: 1, pos: 2245
type: A, layer: 1, pos: 2246
type: A, layer: 1, pos: 2247
type: A, layer: 1, pos: 2248
type: A, layer: 1, pos: 2249
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2639

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 323

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233388, upper bound: 0.1232406
time: 28.51 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233392, upper bound: 0.1232430
time: 151.23 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: 0.0356231, 1.6483717, 0.0322559, 1.6503108, -1.2105045, 1.2130070
1: -3.3674707, -0.8333554, -3.3752551, -0.8312016, -1.6034218, 1.6068734
2: -0.8844246, -0.3122693, -0.8847272, -0.3123001, -0.2169648, 0.2173471
3: -0.1337419, 0.2342571, -0.1349747, 0.2343095, -0.1764506, 0.1774433
4: -2.7804849, -2.2557077, -2.7806239, -2.2556748, -0.2452438, 0.2455342
5: -0.9361289, -0.5459318, -0.9373421, -0.5456775, -0.1956761, 0.1964986
6: -0.5228302, 0.5104322, -0.5227070, 0.5104920, -0.1523802, 0.1525226
7: -1.0608459, 0.0852418, -1.0622630, 0.0860939, -0.5858471, 0.5865643
8: -4.9044352, -3.1234150, -4.9081640, -3.1221709, -0.9021586, 0.9049816
9: -2.8540683, -1.3095710, -2.8593476, -1.3086414, -0.7383940, 0.7415813

Time for backsubstitution: 6.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 262
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2391
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 3054
type: A, layer: 1, pos: 3067
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 2332
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 3052
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 3051
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 2327
type: A, layer: 1, pos: 3249
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 334
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2111
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 3003
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 3009
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 273
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2223
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 2284
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 3430
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 2525
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 2217
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 2965
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 2290
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 2055
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 2960
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 3461
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 2225
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 2233
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 2517
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 2226
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 2995
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 506
type: A, layer: 1, pos: 2480
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 3460
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 2037
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 3031
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 3030
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2310
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2286
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 3246
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 2929
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2144
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2240
type: A, layer: 1, pos: 2241
type: A, layer: 1, pos: 2242
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2244
type: A, layer: 1, pos: 2245
type: A, layer: 1, pos: 2246
type: A, layer: 1, pos: 2247
type: A, layer: 1, pos: 2248
type: A, layer: 1, pos: 2249
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2639

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 323

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233371, upper bound: 0.1232707
time: 8.30 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233375, upper bound: 0.1232696
time: 172.17 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: 0.0324311, 1.6500595, 0.0320504, 1.6494982, -1.2113903, 1.2138203
1: -3.3717155, -0.8327947, -3.3724461, -0.8340340, -1.5990934, 1.6124368
2: -0.8844376, -0.3122886, -0.8844687, -0.3123704, -0.2167825, 0.2171082
3: -0.1347491, 0.2343179, -0.1344333, 0.2343203, -0.1773852, 0.1769439
4: -2.7804093, -2.2557073, -2.7802482, -2.2557049, -0.2453203, 0.2451402
5: -0.9373234, -0.5457896, -0.9369600, -0.5457775, -0.1966491, 0.1958730
6: -0.5226764, 0.5103219, -0.5226567, 0.5103702, -0.1528632, 0.1521909
7: -1.0619780, 0.0855237, -1.0614452, 0.0855476, -0.5867622, 0.5855458
8: -4.9066839, -3.1223109, -4.9069648, -3.1229610, -0.9000531, 0.9057573
9: -2.8560133, -1.3100686, -2.8563840, -1.3109298, -0.7339976, 0.7437779

Time for backsubstitution: 6.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 262
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2391
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 3054
type: A, layer: 1, pos: 3067
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 2332
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 3052
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 3051
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 2327
type: A, layer: 1, pos: 3249
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 334
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2111
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 3003
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 3009
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 273
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2223
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 2284
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 3430
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 2525
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 2217
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 2965
type: A, layer: 1, pos: 2290
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 2055
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 2960
type: A, layer: 1, pos: 3461
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 2225
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 2233
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 2517
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 2226
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 2995
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 506
type: A, layer: 1, pos: 2480
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 3460
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 2037
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 3031
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 3030
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2310
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2286
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 3246
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 2929
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2144
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2240
type: A, layer: 1, pos: 2241
type: A, layer: 1, pos: 2242
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2244
type: A, layer: 1, pos: 2245
type: A, layer: 1, pos: 2246
type: A, layer: 1, pos: 2247
type: A, layer: 1, pos: 2248
type: A, layer: 1, pos: 2249
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2639

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 323

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233359, upper bound: 0.1233049
time: 20.93 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233370, upper bound: 0.1233137
time: 11.03 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: 0.0324190, 1.6506827, 0.0295279, 1.6503108, -1.2124460, 1.2183239
1: -3.3717711, -0.8307443, -3.3790209, -0.8312016, -1.6007243, 1.6217438
2: -0.8844509, -0.3122665, -0.8847213, -0.3122998, -0.2169508, 0.2178045
3: -0.1351483, 0.2343184, -0.1350494, 0.2343591, -0.1777964, 0.1774292
4: -2.7806735, -2.2557065, -2.7806492, -2.2556744, -0.2455900, 0.2455487
5: -0.9375957, -0.5457885, -0.9373821, -0.5455633, -0.1973135, 0.1963850
6: -0.5227121, 0.5103219, -0.5227097, 0.5104012, -0.1529903, 0.1522233
7: -1.0625303, 0.0855253, -1.0622983, 0.0863216, -0.5879605, 0.5863676
8: -4.9067106, -3.1216533, -4.9100800, -3.1221709, -0.9009907, 0.9111086
9: -2.8560238, -1.3084373, -2.8610907, -1.3086410, -0.7348454, 0.7504278

Time for backsubstitution: 6.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 262
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2391
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 3054
type: A, layer: 1, pos: 3067
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 2332
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 3052
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 3051
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 2327
type: A, layer: 1, pos: 3249
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 334
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2111
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 3003
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 3009
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 273
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2223
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 2284
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 3430
type: A, layer: 1, pos: 2525
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 2217
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 2965
type: A, layer: 1, pos: 2290
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2055
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 3461
type: A, layer: 1, pos: 2960
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 2225
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 2233
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 2517
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 2226
type: A, layer: 1, pos: 2995
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 506
type: A, layer: 1, pos: 2480
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 3460
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 2037
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 3031
type: A, layer: 1, pos: 3030
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 2310
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2286
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 3246
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 2929
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2144
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2240
type: A, layer: 1, pos: 2241
type: A, layer: 1, pos: 2242
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2244
type: A, layer: 1, pos: 2245
type: A, layer: 1, pos: 2246
type: A, layer: 1, pos: 2247
type: A, layer: 1, pos: 2248
type: A, layer: 1, pos: 2249
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2639

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 323

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233370, upper bound: 0.1233349
time: 125.49 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233358, upper bound: 0.1233428
time: 6.50 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 138.47 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 138.47
Output dim: 3, lower bound: -0.1233388, upper bound: 0.1232406
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 138.47
Output dim: 3, lower bound: -0.1233392, upper bound: 0.1232430
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 138.47
Output dim: 3, lower bound: -0.1233371, upper bound: 0.1232707
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 138.47
Output dim: 3, lower bound: -0.1233375, upper bound: 0.1232696
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 138.47
Output dim: 3, lower bound: -0.1233359, upper bound: 0.1233049
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 138.47
Output dim: 3, lower bound: -0.1233370, upper bound: 0.1233137
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 138.47
Output dim: 3, lower bound: -0.1233370, upper bound: 0.1233349
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 138.47
Output dim: 3, lower bound: -0.1233358, upper bound: 0.1233428

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0356603, 1.6477280, 0.0347989, 1.6494825, -1.2092924, 1.2083156
1: -3.3673887, -0.8383236, -3.3686602, -0.8362446, -1.5995656, 1.5946487
2: -0.8843634, -0.3124412, -0.8844374, -0.3124839, -0.2164598, 0.2162651
3: -0.1333389, 0.2337663, -0.1343541, 0.2338993, -0.1756978, 0.1765552
4: -2.7794783, -2.2557099, -2.7796588, -2.2557061, -0.2439314, 0.2443344
5: -0.9358537, -0.5466066, -0.9369182, -0.5464115, -0.1944539, 0.1952719
6: -0.5205815, 0.5104318, -0.5209767, 0.5104607, -0.1498857, 0.1506851
7: -1.0602691, 0.0851666, -1.0613911, 0.0852642, -0.5842204, 0.5853233
8: -4.9042597, -3.1240728, -4.9049358, -3.1229601, -0.9010667, 0.8995118
9: -2.8540130, -1.3123634, -2.8546057, -1.3118095, -0.7364929, 0.7335696

Time for backsubstitution: 6.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 262
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2391
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 3067
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 2332
type: B, layer: 1, pos: 2541
type: B, layer: 1, pos: 3052
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 3051
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 2327
type: B, layer: 1, pos: 3249
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 2570
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 2585
type: B, layer: 1, pos: 145
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 334
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2111
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 3003
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 3009
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 273
type: B, layer: 1, pos: 2296
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 2223
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 2510
type: B, layer: 1, pos: 2284
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 2056
type: B, layer: 1, pos: 3430
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 2525
type: B, layer: 1, pos: 2666
type: B, layer: 1, pos: 2217
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2232
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 2965
type: B, layer: 1, pos: 2290
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 2190
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 2055
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 2960
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 3461
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 3462
type: B, layer: 1, pos: 2225
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 2040
type: B, layer: 1, pos: 2353
type: B, layer: 1, pos: 2233
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 2517
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 2226
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 2995
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 2221
type: B, layer: 1, pos: 2036
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 506
type: B, layer: 1, pos: 2480
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 3460
type: B, layer: 1, pos: 2052
type: B, layer: 1, pos: 3446
type: B, layer: 1, pos: 2037
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 3031
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 3030
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2310
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 2547
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 2286
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 3246
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 2929
type: B, layer: 1, pos: 2205
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2234
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2240
type: B, layer: 1, pos: 2241
type: B, layer: 1, pos: 2242
type: B, layer: 1, pos: 2243
type: B, layer: 1, pos: 2244
type: B, layer: 1, pos: 2245
type: B, layer: 1, pos: 2246
type: B, layer: 1, pos: 2247
type: B, layer: 1, pos: 2248
type: B, layer: 1, pos: 2249
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2639

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 262

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233357, upper bound: 0.1230612
time: 19.96 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233363, upper bound: 0.1232409
time: 105.38 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0355656, 1.6477528, 0.0347946, 1.6494951, -1.2098358, 1.2080570
1: -3.3754902, -0.8358293, -3.3686628, -0.8343568, -1.6094602, 1.5952406
2: -0.8850176, -0.3094375, -0.8844396, -0.3123987, -0.2173420, 0.2203809
3: -0.1335304, 0.2342664, -0.1342740, 0.2342634, -0.1754892, 0.1766655
4: -2.7798872, -2.2548785, -2.7798502, -2.2557077, -0.2445039, 0.2467352
5: -0.9368301, -0.5458670, -0.9368758, -0.5458997, -0.1947388, 0.1954567
6: -0.5222220, 0.5164160, -0.5221864, 0.5104609, -0.1502985, 0.1589645
7: -1.0597563, 0.0853167, -1.0613319, 0.0853179, -0.5841781, 0.5852785
8: -4.9030857, -3.1246529, -4.9038696, -3.1229601, -0.9007164, 0.8979366
9: -2.8574116, -1.3113065, -2.8546360, -1.3110292, -0.7408524, 0.7335699

Time for backsubstitution: 6.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 262
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2391
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 3067
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 2332
type: B, layer: 1, pos: 2541
type: B, layer: 1, pos: 3052
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 3051
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 2327
type: B, layer: 1, pos: 3249
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 2570
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 2585
type: B, layer: 1, pos: 145
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 334
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2111
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 3003
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 3009
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 273
type: B, layer: 1, pos: 2296
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 2223
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 2510
type: B, layer: 1, pos: 2284
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 2056
type: B, layer: 1, pos: 2525
type: B, layer: 1, pos: 3430
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 2666
type: B, layer: 1, pos: 2217
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2232
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 2965
type: B, layer: 1, pos: 2290
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 2190
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 2055
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 3461
type: B, layer: 1, pos: 2960
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 3462
type: B, layer: 1, pos: 2225
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 2040
type: B, layer: 1, pos: 2353
type: B, layer: 1, pos: 2233
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 2517
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 2226
type: B, layer: 1, pos: 2995
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 2221
type: B, layer: 1, pos: 2036
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 506
type: B, layer: 1, pos: 2480
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 3460
type: B, layer: 1, pos: 2052
type: B, layer: 1, pos: 2037
type: B, layer: 1, pos: 3446
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 3031
type: B, layer: 1, pos: 3030
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 2310
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 2547
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 2286
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 3246
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 2929
type: B, layer: 1, pos: 2205
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2234
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2240
type: B, layer: 1, pos: 2241
type: B, layer: 1, pos: 2242
type: B, layer: 1, pos: 2243
type: B, layer: 1, pos: 2244
type: B, layer: 1, pos: 2245
type: B, layer: 1, pos: 2246
type: B, layer: 1, pos: 2247
type: B, layer: 1, pos: 2248
type: B, layer: 1, pos: 2249
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2639

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 262

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233366, upper bound: 0.1230624
time: 13.95 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233365, upper bound: 0.1232433
time: 419.04 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0356481, 1.6483512, 0.0322742, 1.6502943, -1.2103480, 1.2128191
1: -3.3674440, -0.8362732, -3.3752346, -0.8334122, -1.6011972, 1.6039538
2: -0.8843762, -0.3124188, -0.8846900, -0.3124126, -0.2166284, 0.2169610
3: -0.1337370, 0.2337667, -0.1349710, 0.2339379, -0.1761090, 0.1770405
4: -2.7797422, -2.2557092, -2.7800605, -2.2556760, -0.2442007, 0.2447427
5: -0.9361260, -0.5466055, -0.9373401, -0.5461971, -0.1951183, 0.1957839
6: -0.5206174, 0.5104321, -0.5210305, 0.5104921, -0.1500129, 0.1507176
7: -1.0608221, 0.0851684, -1.0622437, 0.0860380, -0.5854177, 0.5861458
8: -4.9042869, -3.1234150, -4.9080510, -3.1221707, -0.9020032, 0.9048616
9: -2.8540239, -1.3107324, -2.8593123, -1.3095205, -0.7373413, 0.7402195

Time for backsubstitution: 6.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 262
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2391
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 3067
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 2332
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 2541
type: B, layer: 1, pos: 3052
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 3051
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 2327
type: B, layer: 1, pos: 3249
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 2570
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 2585
type: B, layer: 1, pos: 145
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 334
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2111
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 3003
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 3009
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 273
type: B, layer: 1, pos: 2296
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 2223
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 2510
type: B, layer: 1, pos: 2284
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 2056
type: B, layer: 1, pos: 3430
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 2525
type: B, layer: 1, pos: 2666
type: B, layer: 1, pos: 2217
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2232
type: B, layer: 1, pos: 2965
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 2290
type: B, layer: 1, pos: 2190
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 2055
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 2960
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 3461
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 3462
type: B, layer: 1, pos: 2225
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 2040
type: B, layer: 1, pos: 2353
type: B, layer: 1, pos: 2233
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 2517
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 2226
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 2995
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 2221
type: B, layer: 1, pos: 2036
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 506
type: B, layer: 1, pos: 2480
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 3460
type: B, layer: 1, pos: 2052
type: B, layer: 1, pos: 3446
type: B, layer: 1, pos: 2037
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 3031
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 3030
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2310
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 2547
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 2286
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 3246
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 2929
type: B, layer: 1, pos: 2205
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2234
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2240
type: B, layer: 1, pos: 2241
type: B, layer: 1, pos: 2242
type: B, layer: 1, pos: 2243
type: B, layer: 1, pos: 2244
type: B, layer: 1, pos: 2245
type: B, layer: 1, pos: 2246
type: B, layer: 1, pos: 2247
type: B, layer: 1, pos: 2248
type: B, layer: 1, pos: 2249
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2639

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 262

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233368, upper bound: 0.1230903
time: 11.27 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233365, upper bound: 0.1232703
time: 9.84 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0355515, 1.6483760, 0.0322709, 1.6503084, -1.2108922, 1.2125599
1: -3.3755457, -0.8337803, -3.3752363, -0.8315244, -1.6110920, 1.6045457
2: -0.8850307, -0.3094168, -0.8846923, -0.3123279, -0.2175102, 0.2210830
3: -0.1339267, 0.2342670, -0.1348915, 0.2343022, -0.1759001, 0.1771507
4: -2.7801518, -2.2548783, -2.7802522, -2.2556775, -0.2447737, 0.2471438
5: -0.9371011, -0.5458660, -0.9372976, -0.5456855, -0.1954009, 0.1959688
6: -0.5222585, 0.5164164, -0.5222406, 0.5104918, -0.1504256, 0.1589970
7: -1.0603087, 0.0853189, -1.0621848, 0.0860915, -0.5853753, 0.5861008
8: -4.9031124, -3.1239934, -4.9069848, -3.1221712, -0.9016533, 0.9032867
9: -2.8574238, -1.3096759, -2.8593433, -1.3087406, -0.7417002, 0.7402202

Time for backsubstitution: 6.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 262
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2391
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 3067
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 2332
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 2541
type: B, layer: 1, pos: 3052
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 3051
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 2327
type: B, layer: 1, pos: 3249
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 2570
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 2585
type: B, layer: 1, pos: 145
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 334
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2111
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 3003
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 3009
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 273
type: B, layer: 1, pos: 2296
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 2223
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 2510
type: B, layer: 1, pos: 2284
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 2056
type: B, layer: 1, pos: 3430
type: B, layer: 1, pos: 2525
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 2666
type: B, layer: 1, pos: 2217
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2232
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 2965
type: B, layer: 1, pos: 2290
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 2190
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 2055
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 2960
type: B, layer: 1, pos: 3461
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 3462
type: B, layer: 1, pos: 2225
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 2040
type: B, layer: 1, pos: 2353
type: B, layer: 1, pos: 2233
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 2517
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 2226
type: B, layer: 1, pos: 2995
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 2221
type: B, layer: 1, pos: 2036
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 506
type: B, layer: 1, pos: 2480
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 3460
type: B, layer: 1, pos: 2052
type: B, layer: 1, pos: 2037
type: B, layer: 1, pos: 3446
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 3031
type: B, layer: 1, pos: 3030
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 2310
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 2547
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 2286
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 3246
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 2929
type: B, layer: 1, pos: 2205
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2234
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2240
type: B, layer: 1, pos: 2241
type: B, layer: 1, pos: 2242
type: B, layer: 1, pos: 2243
type: B, layer: 1, pos: 2244
type: B, layer: 1, pos: 2245
type: B, layer: 1, pos: 2246
type: B, layer: 1, pos: 2247
type: B, layer: 1, pos: 2248
type: B, layer: 1, pos: 2249
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2639

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 262

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233358, upper bound: 0.1230917
time: 97.32 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233365, upper bound: 0.1232737
time: 11.73 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0324564, 1.6500387, 0.0320697, 1.6494825, -1.2112341, 1.2136321
1: -3.3716912, -0.8357124, -3.3724267, -0.8362446, -1.5968691, 1.6095163
2: -0.8843892, -0.3124377, -0.8844314, -0.3124835, -0.2164460, 0.2167221
3: -0.1347441, 0.2338276, -0.1344296, 0.2339489, -0.1770435, 0.1765412
4: -2.7796669, -2.2557087, -2.7796850, -2.2557058, -0.2442772, 0.2443489
5: -0.9373206, -0.5464632, -0.9369580, -0.5462972, -0.1960914, 0.1951582
6: -0.5204636, 0.5103215, -0.5209801, 0.5103704, -0.1504959, 0.1503859
7: -1.0619540, 0.0854500, -1.0614266, 0.0854918, -0.5863333, 0.5851269
8: -4.9065351, -3.1223116, -4.9068503, -3.1229601, -0.8998985, 0.9056380
9: -2.8559670, -1.3112302, -2.8563476, -1.3118091, -0.7329450, 0.7424163

Time for backsubstitution: 6.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 262
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2391
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 3067
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 2332
type: B, layer: 1, pos: 2541
type: B, layer: 1, pos: 3052
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 3051
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 2327
type: B, layer: 1, pos: 3249
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 2570
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 2585
type: B, layer: 1, pos: 145
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 334
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2111
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 3003
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 3009
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 273
type: B, layer: 1, pos: 2296
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 2223
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 2510
type: B, layer: 1, pos: 2284
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 2056
type: B, layer: 1, pos: 3430
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 2525
type: B, layer: 1, pos: 2666
type: B, layer: 1, pos: 2217
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2232
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 2965
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 2290
type: B, layer: 1, pos: 2190
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 2055
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 2960
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 3461
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 3462
type: B, layer: 1, pos: 2225
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 2040
type: B, layer: 1, pos: 2353
type: B, layer: 1, pos: 2233
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 2517
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 2226
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 2995
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 2221
type: B, layer: 1, pos: 2036
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 506
type: B, layer: 1, pos: 2480
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 3460
type: B, layer: 1, pos: 2052
type: B, layer: 1, pos: 3446
type: B, layer: 1, pos: 2037
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 3031
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 3030
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2310
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 2547
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 2286
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 3246
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 2929
type: B, layer: 1, pos: 2205
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2234
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2240
type: B, layer: 1, pos: 2241
type: B, layer: 1, pos: 2242
type: B, layer: 1, pos: 2243
type: B, layer: 1, pos: 2244
type: B, layer: 1, pos: 2245
type: B, layer: 1, pos: 2246
type: B, layer: 1, pos: 2247
type: B, layer: 1, pos: 2248
type: B, layer: 1, pos: 2249
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2639

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 262

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233369, upper bound: 0.1231283
time: 91.31 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233363, upper bound: 0.1233106
time: 10.23 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0323606, 1.6500630, 0.0320659, 1.6494954, -1.2117772, 1.2133737
1: -3.3797917, -0.8332191, -3.3724291, -0.8343568, -1.6067634, 1.6101091
2: -0.8850436, -0.3094259, -0.8844337, -0.3123983, -0.2173275, 0.2208554
3: -0.1349312, 0.2343279, -0.1343500, 0.2343131, -0.1768339, 0.1766515
4: -2.7800763, -2.2548778, -2.7798753, -2.2557075, -0.2448501, 0.2467500
5: -0.9382937, -0.5457237, -0.9369160, -0.5457855, -0.1963705, 0.1953433
6: -0.5221043, 0.5163059, -0.5221899, 0.5103701, -0.1509088, 0.1586653
7: -1.0614411, 0.0856006, -1.0613672, 0.0855448, -0.5862906, 0.5850819
8: -4.9053612, -3.1228905, -4.9057846, -3.1229601, -0.8995479, 0.9040627
9: -2.8593667, -1.3101733, -2.8563786, -1.3110287, -0.7373040, 0.7424169

Time for backsubstitution: 6.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 262
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2391
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 3067
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 2332
type: B, layer: 1, pos: 2541
type: B, layer: 1, pos: 3052
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 3051
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 2327
type: B, layer: 1, pos: 3249
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 2570
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 2585
type: B, layer: 1, pos: 145
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 334
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2111
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 3003
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 3009
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 273
type: B, layer: 1, pos: 2296
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 2223
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 2510
type: B, layer: 1, pos: 2284
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 2056
type: B, layer: 1, pos: 3430
type: B, layer: 1, pos: 2525
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 2666
type: B, layer: 1, pos: 2217
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2232
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 2965
type: B, layer: 1, pos: 2290
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 2190
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 2055
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 2960
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 3461
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 3462
type: B, layer: 1, pos: 2225
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 2040
type: B, layer: 1, pos: 2353
type: B, layer: 1, pos: 2233
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 2517
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 2226
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 2995
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 2221
type: B, layer: 1, pos: 2036
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 506
type: B, layer: 1, pos: 2480
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 3460
type: B, layer: 1, pos: 2052
type: B, layer: 1, pos: 2037
type: B, layer: 1, pos: 3446
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 3031
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 3030
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2310
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 2547
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 2286
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 3246
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 2929
type: B, layer: 1, pos: 2205
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2234
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2240
type: B, layer: 1, pos: 2241
type: B, layer: 1, pos: 2242
type: B, layer: 1, pos: 2243
type: B, layer: 1, pos: 2244
type: B, layer: 1, pos: 2245
type: B, layer: 1, pos: 2246
type: B, layer: 1, pos: 2247
type: B, layer: 1, pos: 2248
type: B, layer: 1, pos: 2249
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2639

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 262

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233357, upper bound: 0.1231296
time: 27.81 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233356, upper bound: 0.1233104
time: 145.65 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0324428, 1.6506617, 0.0295465, 1.6502950, -1.2122900, 1.2181363
1: -3.3717461, -0.8336620, -3.3790021, -0.8334122, -1.5985001, 1.6188236
2: -0.8844024, -0.3124152, -0.8846841, -0.3124123, -0.2166144, 0.2174184
3: -0.1351434, 0.2338284, -0.1350458, 0.2339875, -0.1774548, 0.1770264
4: -2.7799306, -2.2557080, -2.7800865, -2.2556756, -0.2445469, 0.2447572
5: -0.9375929, -0.5464621, -0.9373801, -0.5460829, -0.1967557, 0.1956702
6: -0.5204995, 0.5103216, -0.5210336, 0.5104013, -0.1506229, 0.1504184
7: -1.0625063, 0.0854519, -1.0622792, 0.0862656, -0.5875313, 0.5859486
8: -4.9065633, -3.1216531, -4.9099665, -3.1221707, -0.9008359, 0.9109883
9: -2.8559780, -1.3095984, -2.8610554, -1.3095202, -0.7337925, 0.7490665

Time for backsubstitution: 6.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 262
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2391
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 3067
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 2332
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 2541
type: B, layer: 1, pos: 3052
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 3051
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 2327
type: B, layer: 1, pos: 3249
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 2570
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 2585
type: B, layer: 1, pos: 145
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 334
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2111
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 3003
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 3009
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 273
type: B, layer: 1, pos: 2296
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 2223
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 2510
type: B, layer: 1, pos: 2284
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2056
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 3430
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 2525
type: B, layer: 1, pos: 2666
type: B, layer: 1, pos: 2217
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2232
type: B, layer: 1, pos: 2965
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 2290
type: B, layer: 1, pos: 2190
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 2055
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 2960
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 3461
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 3462
type: B, layer: 1, pos: 2225
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 2040
type: B, layer: 1, pos: 2353
type: B, layer: 1, pos: 2233
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 2517
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 2226
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 2995
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 2221
type: B, layer: 1, pos: 2036
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 506
type: B, layer: 1, pos: 2480
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 3460
type: B, layer: 1, pos: 2052
type: B, layer: 1, pos: 3446
type: B, layer: 1, pos: 2037
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 3031
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 3030
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2310
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 2547
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 2286
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 3246
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 2929
type: B, layer: 1, pos: 2205
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2234
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2240
type: B, layer: 1, pos: 2241
type: B, layer: 1, pos: 2242
type: B, layer: 1, pos: 2243
type: B, layer: 1, pos: 2244
type: B, layer: 1, pos: 2245
type: B, layer: 1, pos: 2246
type: B, layer: 1, pos: 2247
type: B, layer: 1, pos: 2248
type: B, layer: 1, pos: 2249
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2639

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 262

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233380, upper bound: 0.1231581
time: 10.26 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233384, upper bound: 0.1233396
time: 8.08 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0323477, 1.6506858, 0.0295424, 1.6503086, -1.2128329, 1.2178769
1: -3.3798463, -0.8311696, -3.3790040, -0.8315248, -1.6083939, 1.6194160
2: -0.8850572, -0.3094054, -0.8846862, -0.3123277, -0.2174955, 0.2215576
3: -0.1353287, 0.2343284, -0.1349668, 0.2343517, -0.1772447, 0.1771367
4: -2.7803409, -2.2548769, -2.7802773, -2.2556770, -0.2451202, 0.2471584
5: -0.9385645, -0.5457224, -0.9373379, -0.5455712, -0.1970330, 0.1958552
6: -0.5221403, 0.5163060, -0.5222436, 0.5104010, -0.1510360, 0.1586978
7: -1.0619944, 0.0856023, -1.0622201, 0.0863193, -0.5874884, 0.5859040
8: -4.9053888, -3.1222320, -4.9088998, -3.1221712, -0.9004847, 0.9094135
9: -2.8593776, -1.3085430, -2.8610864, -1.3087406, -0.7381516, 0.7490671

Time for backsubstitution: 6.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 262
type: B, layer: 1, pos: 2137
type: B, layer: 1, pos: 2391
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 3067
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 2332
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 2541
type: B, layer: 1, pos: 3052
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 3051
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 2556
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 2327
type: B, layer: 1, pos: 3249
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 2570
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 2585
type: B, layer: 1, pos: 145
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 334
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2111
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 3003
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 3009
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 273
type: B, layer: 1, pos: 2296
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 2223
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 2510
type: B, layer: 1, pos: 2284
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 2056
type: B, layer: 1, pos: 3430
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 2525
type: B, layer: 1, pos: 2666
type: B, layer: 1, pos: 2217
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2232
type: B, layer: 1, pos: 2965
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 2290
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 2190
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 2055
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 2960
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 3461
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 3462
type: B, layer: 1, pos: 2225
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 2040
type: B, layer: 1, pos: 2353
type: B, layer: 1, pos: 2233
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 2517
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 2226
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 2995
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 2221
type: B, layer: 1, pos: 2036
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 506
type: B, layer: 1, pos: 2480
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 3460
type: B, layer: 1, pos: 2052
type: B, layer: 1, pos: 2037
type: B, layer: 1, pos: 3446
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 3031
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 3030
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2310
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 2547
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 2948
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 2286
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 3246
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 2929
type: B, layer: 1, pos: 2205
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2234
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2240
type: B, layer: 1, pos: 2241
type: B, layer: 1, pos: 2242
type: B, layer: 1, pos: 2243
type: B, layer: 1, pos: 2244
type: B, layer: 1, pos: 2245
type: B, layer: 1, pos: 2246
type: B, layer: 1, pos: 2247
type: B, layer: 1, pos: 2248
type: B, layer: 1, pos: 2249
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2639

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 262

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233360, upper bound: 0.1231575
time: 145.45 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233373, upper bound: 0.1233425
time: 11.67 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 163.72 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 163.72
Output dim: 3, lower bound: -0.1233357, upper bound: 0.1230612
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 163.72
Output dim: 3, lower bound: -0.1233363, upper bound: 0.1232409
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 163.72
Output dim: 3, lower bound: -0.1233366, upper bound: 0.1230624
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 163.72
Output dim: 3, lower bound: -0.1233365, upper bound: 0.1232433
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 163.72
Output dim: 3, lower bound: -0.1233368, upper bound: 0.1230903
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 163.72
Output dim: 3, lower bound: -0.1233365, upper bound: 0.1232703
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 163.72
Output dim: 3, lower bound: -0.1233358, upper bound: 0.1230917
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 163.72
Output dim: 3, lower bound: -0.1233365, upper bound: 0.1232737
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 163.72
Output dim: 3, lower bound: -0.1233369, upper bound: 0.1231283
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 163.72
Output dim: 3, lower bound: -0.1233363, upper bound: 0.1233106
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 163.72
Output dim: 3, lower bound: -0.1233357, upper bound: 0.1231296
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 163.72
Output dim: 3, lower bound: -0.1233356, upper bound: 0.1233104
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 163.72
Output dim: 3, lower bound: -0.1233380, upper bound: 0.1231581
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 163.72
Output dim: 3, lower bound: -0.1233384, upper bound: 0.1233396
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 163.72
Output dim: 3, lower bound: -0.1233360, upper bound: 0.1231575
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 163.72
Output dim: 3, lower bound: -0.1233373, upper bound: 0.1233425

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0358448, 1.6425686, 0.0360270, 1.6423273, -1.2020268, 1.2020121
1: -3.3670020, -0.8383384, -3.3680794, -0.8361931, -1.5993539, 1.5941143
2: -0.8841261, -0.3140568, -0.8837869, -0.3147846, -0.2140562, 0.2141086
3: -0.1310087, 0.2337320, -0.1310481, 0.2333931, -0.1728369, 0.1731474
4: -2.7793305, -2.2579291, -2.7790871, -2.2588739, -0.2406070, 0.2415426
5: -0.9333851, -0.5466181, -0.9334207, -0.5468946, -0.1916189, 0.1918368
6: -0.5170066, 0.5104269, -0.5158874, 0.5095849, -0.1454673, 0.1456029
7: -1.0595294, 0.0831091, -1.0606139, 0.0826752, -0.5804250, 0.5819105
8: -4.9042306, -3.1274586, -4.9041052, -3.1277809, -0.8963423, 0.8954607
9: -2.8539906, -1.3126092, -2.8546300, -1.3121481, -0.7360225, 0.7331675

Time for backsubstitution: 6.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2391
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 3054
type: A, layer: 1, pos: 3067
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 2332
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 3052
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 3051
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 262
type: A, layer: 1, pos: 2327
type: A, layer: 1, pos: 3249
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 334
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2111
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 3003
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 3009
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 273
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2223
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 2284
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 3430
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 2525
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 2217
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 2965
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 2290
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 2055
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2960
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 3461
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 2225
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 2233
type: A, layer: 1, pos: 2517
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 2226
type: A, layer: 1, pos: 2995
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 506
type: A, layer: 1, pos: 2480
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 3460
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 2037
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 3031
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 3030
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2310
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2286
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 3246
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 2929
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2144
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2240
type: A, layer: 1, pos: 2241
type: A, layer: 1, pos: 2242
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2244
type: A, layer: 1, pos: 2245
type: A, layer: 1, pos: 2246
type: A, layer: 1, pos: 2247
type: A, layer: 1, pos: 2248
type: A, layer: 1, pos: 2249
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2639

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2137

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1232306, upper bound: 0.1230502
time: 12.94 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233335, upper bound: 0.1230549
time: 140.85 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0356617, 1.6477165, 0.0347984, 1.6494629, -1.2043033, 1.2083032
1: -3.3673725, -0.8383231, -3.3686337, -0.8362460, -1.5994409, 1.5949244
2: -0.8843554, -0.3124437, -0.8844247, -0.3124879, -0.2144374, 0.2162379
3: -0.1333368, 0.2337662, -0.1343507, 0.2338994, -0.1756938, 0.1739186
4: -2.7794774, -2.2557149, -2.7796576, -2.2557139, -0.2411930, 0.2443298
5: -0.9358498, -0.5466068, -0.9369115, -0.5464116, -0.1944461, 0.1924083
6: -0.5205806, 0.5104315, -0.5209761, 0.5104604, -0.1498839, 0.1451041
7: -1.0602671, 0.0851377, -1.0613868, 0.0852178, -0.5828402, 0.5853188
8: -4.9042587, -3.1240773, -4.9049349, -3.1229665, -0.8963842, 0.8995025
9: -2.8540127, -1.3123698, -2.8546045, -1.3118210, -0.7360537, 0.7335679

Time for backsubstitution: 6.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2391
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 3054
type: A, layer: 1, pos: 3067
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 2332
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 3052
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 3051
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 262
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 2327
type: A, layer: 1, pos: 3249
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 334
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2111
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 3003
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 3009
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 273
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2223
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 2284
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 3430
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 2525
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 2217
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 2965
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 2290
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 2055
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 2960
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 3461
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 2225
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 2233
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 2517
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 2226
type: A, layer: 1, pos: 2995
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 506
type: A, layer: 1, pos: 2480
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 3460
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 2037
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 3031
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 3030
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2310
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2286
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 3246
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 2929
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2144
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2240
type: A, layer: 1, pos: 2241
type: A, layer: 1, pos: 2242
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2244
type: A, layer: 1, pos: 2245
type: A, layer: 1, pos: 2246
type: A, layer: 1, pos: 2247
type: A, layer: 1, pos: 2248
type: A, layer: 1, pos: 2249
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2639

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2137

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1232331, upper bound: 0.1232298
time: 12.87 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233341, upper bound: 0.1232410
time: 15.58 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0357482, 1.6425927, 0.0360236, 1.6423402, -1.2025695, 1.2017535
1: -3.3751023, -0.8358445, -3.3680811, -0.8343053, -1.6092485, 1.5947063
2: -0.8847808, -0.3110545, -0.8837897, -0.3146997, -0.2149389, 0.2182238
3: -0.1312033, 0.2342319, -0.1309682, 0.2337571, -0.1726294, 0.1732578
4: -2.7797382, -2.2570982, -2.7792783, -2.2588758, -0.2411788, 0.2439436
5: -0.9343637, -0.5458785, -0.9333784, -0.5463830, -0.1919082, 0.1920217
6: -0.5186477, 0.5164114, -0.5170969, 0.5095848, -0.1458800, 0.1538823
7: -1.0590148, 0.0832590, -1.0605547, 0.0827288, -0.5803828, 0.5818660
8: -4.9030571, -3.1280375, -4.9030390, -3.1277812, -0.8959916, 0.8938854
9: -2.8573902, -1.3115535, -2.8546610, -1.3113689, -0.7403810, 0.7331678

Time for backsubstitution: 6.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2391
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 3054
type: A, layer: 1, pos: 3067
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 2332
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 3052
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 3051
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 262
type: A, layer: 1, pos: 2327
type: A, layer: 1, pos: 3249
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 334
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 2111
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 3003
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 3009
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 273
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2223
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 2284
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 3430
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 2525
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 2217
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 2965
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 2290
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 2055
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 2960
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 3461
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 2225
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 2233
type: A, layer: 1, pos: 2517
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 2226
type: A, layer: 1, pos: 2995
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 506
type: A, layer: 1, pos: 2480
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 2037
type: A, layer: 1, pos: 3460
type: A, layer: 1, pos: 3031
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 3030
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2310
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2286
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 3246
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 2929
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2144
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2240
type: A, layer: 1, pos: 2241
type: A, layer: 1, pos: 2242
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2244
type: A, layer: 1, pos: 2245
type: A, layer: 1, pos: 2246
type: A, layer: 1, pos: 2247
type: A, layer: 1, pos: 2248
type: A, layer: 1, pos: 2249
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2639

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2137

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1232340, upper bound: 0.1230521
time: 74.40 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233335, upper bound: 0.1230601
time: 14.22 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0355656, 1.6477413, 0.0347955, 1.6494763, -1.2048470, 1.2080442
1: -3.3754745, -0.8358293, -3.3686371, -0.8343577, -1.6093352, 1.5955172
2: -0.8850098, -0.3094396, -0.8844271, -0.3124027, -0.2153195, 0.2203528
3: -0.1335284, 0.2342664, -0.1342705, 0.2342634, -0.1754854, 0.1740288
4: -2.7798858, -2.2548840, -2.7798483, -2.2557158, -0.2417655, 0.2467304
5: -0.9368260, -0.5458671, -0.9368692, -0.5458998, -0.1947310, 0.1925931
6: -0.5222217, 0.5164160, -0.5221853, 0.5104601, -0.1502966, 0.1533835
7: -1.0597539, 0.0852880, -1.0613279, 0.0852713, -0.5827981, 0.5852739
8: -4.9030848, -3.1246564, -4.9038687, -3.1229668, -0.8960340, 0.8979276
9: -2.8574126, -1.3113143, -2.8546355, -1.3110414, -0.7404127, 0.7335681

Time for backsubstitution: 6.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2391
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 3054
type: A, layer: 1, pos: 3067
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 2332
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 3052
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 3051
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 262
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 2327
type: A, layer: 1, pos: 3249
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 334
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 2111
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 3003
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 3009
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 273
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2223
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 2284
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 3430
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 2525
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 2217
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 2965
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 2290
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 2055
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 2960
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 3461
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 2225
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 2233
type: A, layer: 1, pos: 2517
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 2226
type: A, layer: 1, pos: 2995
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 506
type: A, layer: 1, pos: 2480
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 2037
type: A, layer: 1, pos: 3460
type: A, layer: 1, pos: 3031
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 3030
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2310
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2286
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 3246
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 2929
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2144
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2240
type: A, layer: 1, pos: 2241
type: A, layer: 1, pos: 2242
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2244
type: A, layer: 1, pos: 2245
type: A, layer: 1, pos: 2246
type: A, layer: 1, pos: 2247
type: A, layer: 1, pos: 2248
type: A, layer: 1, pos: 2249
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2639

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2137

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1232329, upper bound: 0.1232321
time: 101.70 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233323, upper bound: 0.1232423
time: 11.63 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0358310, 1.6431911, 0.0335038, 1.6431394, -1.2030827, 1.2065126
1: -3.3670571, -0.8362889, -3.3746538, -0.8333607, -1.6009851, 1.6034189
2: -0.8841392, -0.3140348, -0.8840397, -0.3147139, -0.2142247, 0.2148044
3: -0.1314061, 0.2337326, -0.1316639, 0.2334317, -0.1732481, 0.1736327
4: -2.7795944, -2.2579284, -2.7794886, -2.2588437, -0.2408755, 0.2419503
5: -0.9336574, -0.5466169, -0.9338427, -0.5466804, -0.1922835, 0.1923488
6: -0.5170428, 0.5104268, -0.5159409, 0.5096158, -0.1455944, 0.1456354
7: -1.0600822, 0.0831106, -1.0614663, 0.0834486, -0.5816224, 0.5827316
8: -4.9042587, -3.1267996, -4.9072189, -3.1269906, -0.8972790, 0.9008111
9: -2.8540020, -1.3109787, -2.8593364, -1.3098598, -0.7368703, 0.7398177

Time for backsubstitution: 6.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2391
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 3054
type: A, layer: 1, pos: 3067
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 2332
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 3052
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 3051
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 262
type: A, layer: 1, pos: 2327
type: A, layer: 1, pos: 3249
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 334
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2111
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 3003
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 3009
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 273
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2223
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 2284
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 3430
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 2525
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 2217
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 2965
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 2290
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 2055
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 2960
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 3461
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 2225
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 2233
type: A, layer: 1, pos: 2517
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 2226
type: A, layer: 1, pos: 2995
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 506
type: A, layer: 1, pos: 2480
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 3460
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 2037
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 3031
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 3030
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2310
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2286
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 3246
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 2929
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2144
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2240
type: A, layer: 1, pos: 2241
type: A, layer: 1, pos: 2242
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2244
type: A, layer: 1, pos: 2245
type: A, layer: 1, pos: 2246
type: A, layer: 1, pos: 2247
type: A, layer: 1, pos: 2248
type: A, layer: 1, pos: 2249
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2639

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2137

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1232330, upper bound: 0.1230800
time: 8.23 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233329, upper bound: 0.1230855
time: 12.49 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0356476, 1.6483393, 0.0322752, 1.6502752, -1.2053593, 1.2128060
1: -3.3674278, -0.8362737, -3.3752098, -0.8334131, -1.6010723, 1.6042290
2: -0.8843683, -0.3124213, -0.8846775, -0.3124167, -0.2146062, 0.2169336
3: -0.1337348, 0.2337669, -0.1349675, 0.2339378, -0.1761051, 0.1744039
4: -2.7797406, -2.2557142, -2.7800586, -2.2556839, -0.2414621, 0.2447378
5: -0.9361219, -0.5466053, -0.9373336, -0.5461973, -0.1951105, 0.1929203
6: -0.5206168, 0.5104321, -0.5210297, 0.5104914, -0.1500109, 0.1451366
7: -1.0608197, 0.0851397, -1.0622399, 0.0859917, -0.5840377, 0.5861406
8: -4.9042873, -3.1234190, -4.9080505, -3.1221766, -0.8973211, 0.9048529
9: -2.8540235, -1.3107402, -2.8593125, -1.3095319, -0.7369018, 0.7402182

Time for backsubstitution: 6.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2391
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 3054
type: A, layer: 1, pos: 3067
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 2332
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 3052
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 3051
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 262
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 2327
type: A, layer: 1, pos: 3249
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 334
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2111
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 3003
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 3009
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 273
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2223
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 2284
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 3430
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 2525
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 2217
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 2965
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 2290
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 2055
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 2960
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 3461
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 2225
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 2233
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 2517
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 2226
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 2995
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 506
type: A, layer: 1, pos: 2480
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 3460
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 2037
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 3031
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 3030
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2310
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2286
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 3246
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 2929
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2144
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2240
type: A, layer: 1, pos: 2241
type: A, layer: 1, pos: 2242
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2244
type: A, layer: 1, pos: 2245
type: A, layer: 1, pos: 2246
type: A, layer: 1, pos: 2247
type: A, layer: 1, pos: 2248
type: A, layer: 1, pos: 2249
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2639

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2137

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1232316, upper bound: 0.1232571
time: 439.65 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233344, upper bound: 0.1232670
time: 67.92 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0357342, 1.6432145, 0.0335009, 1.6431532, -1.2036254, 1.2062541
1: -3.3751571, -0.8337946, -3.3746562, -0.8314724, -1.6108789, 1.6040117
2: -0.8847940, -0.3110338, -0.8840423, -0.3146290, -0.2151071, 0.2189257
3: -0.1315991, 0.2342325, -0.1315848, 0.2337957, -0.1730400, 0.1737429
4: -2.7800028, -2.2570977, -2.7796798, -2.2588453, -0.2414482, 0.2443514
5: -0.9346347, -0.5458776, -0.9338007, -0.5461687, -0.1925702, 0.1925337
6: -0.5186841, 0.5164115, -0.5171508, 0.5096158, -0.1460071, 0.1539148
7: -1.0595675, 0.0832612, -1.0614079, 0.0835030, -0.5815804, 0.5826874
8: -4.9030857, -3.1273789, -4.9061537, -3.1269917, -0.8969288, 0.8992356
9: -2.8574014, -1.3099222, -2.8593678, -1.3090799, -0.7412295, 0.7398183

Time for backsubstitution: 6.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2391
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 3054
type: A, layer: 1, pos: 3067
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 2332
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 3052
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 3051
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 262
type: A, layer: 1, pos: 2327
type: A, layer: 1, pos: 3249
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 334
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 2111
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 3003
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 3009
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 273
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2223
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 2284
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 3430
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 2525
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 2217
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 2965
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 2290
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 2055
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 2960
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 3461
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 2225
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 2233
type: A, layer: 1, pos: 2517
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 2226
type: A, layer: 1, pos: 2995
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 506
type: A, layer: 1, pos: 2480
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 2037
type: A, layer: 1, pos: 3460
type: A, layer: 1, pos: 3031
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 3030
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2310
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2286
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 3246
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 2929
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2144
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2240
type: A, layer: 1, pos: 2241
type: A, layer: 1, pos: 2242
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2244
type: A, layer: 1, pos: 2245
type: A, layer: 1, pos: 2246
type: A, layer: 1, pos: 2247
type: A, layer: 1, pos: 2248
type: A, layer: 1, pos: 2249
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2639

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2137

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1232303, upper bound: 0.1230506
time: 78.25 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233323, upper bound: 0.1230890
time: 10.52 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0355518, 1.6483634, 0.0322711, 1.6502893, -1.2059025, 1.2125475
1: -3.3755298, -0.8337803, -3.3752127, -0.8315244, -1.6109672, 1.6048213
2: -0.8850228, -0.3094192, -0.8846798, -0.3123323, -0.2154878, 0.2210549
3: -0.1339247, 0.2342671, -0.1348880, 0.2343021, -0.1758962, 0.1745141
4: -2.7801504, -2.2548835, -2.7802501, -2.2556856, -0.2420352, 0.2471390
5: -0.9370970, -0.5458659, -0.9372910, -0.5456855, -0.1953930, 0.1931052
6: -0.5222582, 0.5164161, -0.5222396, 0.5104915, -0.1504237, 0.1534160
7: -1.0603062, 0.0852900, -1.0621809, 0.0860456, -0.5839956, 0.5860963
8: -4.9031134, -3.1239977, -4.9069834, -3.1221771, -0.8969711, 0.9032776
9: -2.8574233, -1.3096831, -2.8593431, -1.3087523, -0.7412604, 0.7402190

Time for backsubstitution: 6.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2391
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 3054
type: A, layer: 1, pos: 3067
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 2332
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 3052
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 3051
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 262
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 2327
type: A, layer: 1, pos: 3249
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 334
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 2111
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 3003
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 3009
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 273
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2223
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 2284
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 3430
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 2525
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 2217
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 2965
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 2290
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 2055
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 2960
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 3461
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 2225
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 2233
type: A, layer: 1, pos: 2517
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 2226
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 2995
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 506
type: A, layer: 1, pos: 2480
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 2037
type: A, layer: 1, pos: 3460
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 3031
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 3030
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2310
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2286
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 3246
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 2929
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2144
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2240
type: A, layer: 1, pos: 2241
type: A, layer: 1, pos: 2242
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2244
type: A, layer: 1, pos: 2245
type: A, layer: 1, pos: 2246
type: A, layer: 1, pos: 2247
type: A, layer: 1, pos: 2248
type: A, layer: 1, pos: 2249
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2639

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2137

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1232338, upper bound: 0.1232431
time: 153.39 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233326, upper bound: 0.1232708
time: 13.30 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0326400, 1.6448779, 0.0332987, 1.6423278, -1.2039676, 1.2073287
1: -3.3713019, -0.8357286, -3.3718452, -0.8361921, -1.5966563, 1.6089802
2: -0.8841523, -0.3140542, -0.8837810, -0.3147842, -0.2140430, 0.2145663
3: -0.1324129, 0.2337933, -0.1311234, 0.2334427, -0.1741824, 0.1731334
4: -2.7795193, -2.2579284, -2.7791128, -2.2588737, -0.2409529, 0.2415571
5: -0.9348516, -0.5464748, -0.9334604, -0.5467804, -0.1932565, 0.1917230
6: -0.5168886, 0.5103161, -0.5158905, 0.5094938, -0.1460775, 0.1453036
7: -1.0612135, 0.0833930, -1.0606489, 0.0829022, -0.5825372, 0.5817132
8: -4.9065056, -3.1256967, -4.9060197, -3.1277809, -0.8951740, 0.9015857
9: -2.8559444, -1.3114762, -2.8563735, -1.3121488, -0.7324740, 0.7420151

Time for backsubstitution: 6.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2391
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 3054
type: A, layer: 1, pos: 3067
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 2332
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 3052
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 3051
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 262
type: A, layer: 1, pos: 2327
type: A, layer: 1, pos: 3249
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 334
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2111
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 3003
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 3009
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 273
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2223
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 2284
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 3430
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 2525
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 2217
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 2965
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 2290
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 2055
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 2960
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 3461
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 2225
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 2233
type: A, layer: 1, pos: 2517
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 2226
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 2995
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 506
type: A, layer: 1, pos: 2480
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 3460
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 2037
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 3031
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 3030
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2310
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2286
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 3246
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 2929
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2144
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2240
type: A, layer: 1, pos: 2241
type: A, layer: 1, pos: 2242
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2244
type: A, layer: 1, pos: 2245
type: A, layer: 1, pos: 2246
type: A, layer: 1, pos: 2247
type: A, layer: 1, pos: 2248
type: A, layer: 1, pos: 2249
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2639

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2137

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1232324, upper bound: 0.1231180
time: 8.63 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233341, upper bound: 0.1231277
time: 6.88 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0324554, 1.6500273, 0.0320697, 1.6494634, -1.2062453, 1.2136198
1: -3.3716741, -0.8357129, -3.3724008, -0.8362446, -1.5967443, 1.6097931
2: -0.8843811, -0.3124406, -0.8844188, -0.3124875, -0.2144237, 0.2166948
3: -0.1347420, 0.2338277, -0.1344262, 0.2339488, -0.1770397, 0.1739045
4: -2.7796657, -2.2557135, -2.7796834, -2.2557142, -0.2415388, 0.2443442
5: -0.9373165, -0.5464633, -0.9369515, -0.5462973, -0.1960835, 0.1922946
6: -0.5204627, 0.5103211, -0.5209789, 0.5103697, -0.1504939, 0.1448049
7: -1.0619514, 0.0854210, -1.0614229, 0.0854452, -0.5849531, 0.5851223
8: -4.9065332, -3.1223161, -4.9068499, -3.1229670, -0.8952167, 0.9056290
9: -2.8559666, -1.3112366, -2.8563483, -1.3118207, -0.7325053, 0.7424150

Time for backsubstitution: 6.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2137
type: A, layer: 1, pos: 2391
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 3054
type: A, layer: 1, pos: 3067
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 2332
type: A, layer: 1, pos: 2541
type: A, layer: 1, pos: 3052
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 3051
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 262
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 2556
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 2327
type: A, layer: 1, pos: 3249
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 334
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2111
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 3003
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 3009
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 273
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2223
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 2284
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 3430
type: A, layer: 1, pos: 2525
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 2217
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 2965
type: A, layer: 1, pos: 2290
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 2055
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 2960
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 3461
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 2225
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 2233
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 2517
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 2226
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 2995
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 506
type: A, layer: 1, pos: 2480
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 3460
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 2037
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 3031
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 3030
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2310
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 2948
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2286
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 3246
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 2929
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2144
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2240
type: A, layer: 1, pos: 2241
type: A, layer: 1, pos: 2242
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2244
type: A, layer: 1, pos: 2245
type: A, layer: 1, pos: 2246
type: A, layer: 1, pos: 2247
type: A, layer: 1, pos: 2248
type: A, layer: 1, pos: 2249
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2639

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 2137

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1232334, upper bound: 0.1232978
time: 217.09 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233337, upper bound: 0.1233051
time: 186.36 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 71.32 + 3775.73 = 3847.05 seconds
