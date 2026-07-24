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
execution time: IAR + RelationalAnalysis = 7.42 + 63.85 = 71.27 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.1234413, upper bound: 0.1234429

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 3009
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2480
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 572
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 2965

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2172

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1234255, upper bound: 0.1234297
time: 32.93 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1234266, upper bound: 0.1234289
time: 80.70 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 113.64 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 113.64
Output dim: 3, lower bound: -0.1234255, upper bound: 0.1234297
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 113.64
Output dim: 3, lower bound: -0.1234266, upper bound: 0.1234289

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 0.0302720, 1.6531119, 0.0302720, 1.6531119, -1.2185150, 1.2184868
1: -3.3763952, -0.8267784, -3.3763952, -0.8267784, -1.6202102, 1.6201664
2: -0.8847626, -0.3119067, -0.8847626, -0.3119067, -0.2177581, 0.2177579
3: -0.1364306, 0.2343346, -0.1364306, 0.2343346, -0.1792384, 0.1792383
4: -2.7811904, -2.2556832, -2.7811904, -2.2556832, -0.2461713, 0.2461712
5: -0.9392171, -0.5457087, -0.9392171, -0.5457087, -0.1986741, 0.1986714
6: -0.5227640, 0.5105959, -0.5227640, 0.5105959, -0.1530267, 0.1530260
7: -1.0645843, 0.0856791, -1.0645843, 0.0856791, -0.5892586, 0.5892816
8: -4.9083729, -3.1190684, -4.9083729, -3.1190684, -0.9091890, 0.9091079
9: -2.8582137, -1.3061166, -2.8582137, -1.3061166, -0.7476203, 0.7476068

Time for backsubstitution: 5.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 572
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2480
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 3009
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2290

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 885

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1234246, upper bound: 0.1234267
time: 48.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1234246, upper bound: 0.1234300
time: 13.52 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 0.0302720, 1.6531119, 0.0302720, 1.6531119, -1.2184868, 1.2185149
1: -3.3763952, -0.8267784, -3.3763952, -0.8267784, -1.6201663, 1.6202103
2: -0.8847626, -0.3119067, -0.8847626, -0.3119067, -0.2177579, 0.2177581
3: -0.1364306, 0.2343346, -0.1364306, 0.2343346, -0.1792383, 0.1792385
4: -2.7811904, -2.2556832, -2.7811904, -2.2556832, -0.2461713, 0.2461713
5: -0.9392171, -0.5457087, -0.9392171, -0.5457087, -0.1986714, 0.1986741
6: -0.5227640, 0.5105959, -0.5227640, 0.5105959, -0.1530260, 0.1530267
7: -1.0645843, 0.0856791, -1.0645843, 0.0856791, -0.5892817, 0.5892586
8: -4.9083729, -3.1190684, -4.9083729, -3.1190684, -0.9091079, 0.9091890
9: -2.8582137, -1.3061166, -2.8582137, -1.3061166, -0.7476068, 0.7476203

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 2480
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 3009
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 572
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 821

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2038

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1234246, upper bound: 0.1234263
time: 11.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1234239, upper bound: 0.1234267
time: 59.97 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 77.75 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 77.75
Output dim: 3, lower bound: -0.1234246, upper bound: 0.1234267
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 77.75
Output dim: 3, lower bound: -0.1234246, upper bound: 0.1234300
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 77.75
Output dim: 3, lower bound: -0.1234246, upper bound: 0.1234263
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 77.75
Output dim: 3, lower bound: -0.1234239, upper bound: 0.1234267

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0302720, 1.6531119, 0.0302720, 1.6531119, -1.2185150, 1.2184868
1: -3.3763952, -0.8267784, -3.3763952, -0.8267784, -1.6202102, 1.6201664
2: -0.8847626, -0.3119067, -0.8847626, -0.3119067, -0.2177581, 0.2177579
3: -0.1364306, 0.2343346, -0.1364306, 0.2343346, -0.1792384, 0.1792383
4: -2.7811904, -2.2556832, -2.7811904, -2.2556832, -0.2461713, 0.2461712
5: -0.9392171, -0.5457087, -0.9392171, -0.5457087, -0.1986741, 0.1986714
6: -0.5227640, 0.5105959, -0.5227640, 0.5105959, -0.1530267, 0.1530260
7: -1.0645843, 0.0856791, -1.0645843, 0.0856791, -0.5892586, 0.5892816
8: -4.9083729, -3.1190684, -4.9083729, -3.1190684, -0.9091890, 0.9091079
9: -2.8582137, -1.3061166, -2.8582137, -1.3061166, -0.7476203, 0.7476068

Time for backsubstitution: 6.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 3009
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 572
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2480
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2174

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 896

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1234232, upper bound: 0.1234279
time: 8.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1234232, upper bound: 0.1234256
time: 38.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0302720, 1.6531119, 0.0302720, 1.6531119, -1.2185150, 1.2184868
1: -3.3763952, -0.8267784, -3.3763952, -0.8267784, -1.6202102, 1.6201664
2: -0.8847626, -0.3119067, -0.8847626, -0.3119067, -0.2177581, 0.2177579
3: -0.1364306, 0.2343346, -0.1364306, 0.2343346, -0.1792384, 0.1792383
4: -2.7811904, -2.2556832, -2.7811904, -2.2556832, -0.2461713, 0.2461712
5: -0.9392171, -0.5457087, -0.9392171, -0.5457087, -0.1986741, 0.1986714
6: -0.5227640, 0.5105959, -0.5227640, 0.5105959, -0.1530267, 0.1530260
7: -1.0645843, 0.0856791, -1.0645843, 0.0856791, -0.5892586, 0.5892816
8: -4.9083729, -3.1190684, -4.9083729, -3.1190684, -0.9091890, 0.9091079
9: -2.8582137, -1.3061166, -2.8582137, -1.3061166, -0.7476203, 0.7476068

Time for backsubstitution: 6.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 572
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3009
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 2480
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 778

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3246

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233834, upper bound: 0.1233836
time: 13.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233807, upper bound: 0.1233885
time: 8.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0302720, 1.6531119, 0.0302720, 1.6531119, -1.2184477, 1.2184687
1: -3.3763952, -0.8267784, -3.3763952, -0.8267784, -1.6201580, 1.6202024
2: -0.8847626, -0.3119067, -0.8847626, -0.3119067, -0.2177260, 0.2177300
3: -0.1364306, 0.2343346, -0.1364306, 0.2343346, -0.1792373, 0.1792376
4: -2.7811904, -2.2556832, -2.7811904, -2.2556832, -0.2461652, 0.2461650
5: -0.9392171, -0.5457087, -0.9392171, -0.5457087, -0.1986577, 0.1986580
6: -0.5227640, 0.5105959, -0.5227640, 0.5105959, -0.1530124, 0.1530152
7: -1.0645843, 0.0856791, -1.0645843, 0.0856791, -0.5892010, 0.5891631
8: -4.9083729, -3.1190684, -4.9083729, -3.1190684, -0.9090695, 0.9091434
9: -2.8582137, -1.3061166, -2.8582137, -1.3061166, -0.7476058, 0.7476190

Time for backsubstitution: 6.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 3009
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2480
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 572
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2229

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2517

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1234215, upper bound: 0.1234262
time: 12.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1234228, upper bound: 0.1234251
time: 99.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0302720, 1.6531119, 0.0302720, 1.6531119, -1.2184405, 1.2184761
1: -3.3763952, -0.8267784, -3.3763952, -0.8267784, -1.6201587, 1.6202017
2: -0.8847626, -0.3119067, -0.8847626, -0.3119067, -0.2177298, 0.2177263
3: -0.1364306, 0.2343346, -0.1364306, 0.2343346, -0.1792375, 0.1792375
4: -2.7811904, -2.2556832, -2.7811904, -2.2556832, -0.2461649, 0.2461653
5: -0.9392171, -0.5457087, -0.9392171, -0.5457087, -0.1986553, 0.1986604
6: -0.5227640, 0.5105959, -0.5227640, 0.5105959, -0.1530145, 0.1530130
7: -1.0645843, 0.0856791, -1.0645843, 0.0856791, -0.5891862, 0.5891780
8: -4.9083729, -3.1190684, -4.9083729, -3.1190684, -0.9090624, 0.9091506
9: -2.8582137, -1.3061166, -2.8582137, -1.3061166, -0.7476053, 0.7476196

Time for backsubstitution: 6.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 572
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2480
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 3009
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 756

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2666

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1234225, upper bound: 0.1234167
time: 8.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1234139, upper bound: 0.1234147
time: 113.43 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 128.31 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 128.31
Output dim: 3, lower bound: -0.1234232, upper bound: 0.1234279
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 128.31
Output dim: 3, lower bound: -0.1234232, upper bound: 0.1234256
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 128.31
Output dim: 3, lower bound: -0.1233834, upper bound: 0.1233836
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 128.31
Output dim: 3, lower bound: -0.1233807, upper bound: 0.1233885
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 128.31
Output dim: 3, lower bound: -0.1234215, upper bound: 0.1234262
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 128.31
Output dim: 3, lower bound: -0.1234228, upper bound: 0.1234251
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 128.31
Output dim: 3, lower bound: -0.1234225, upper bound: 0.1234167
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 128.31
Output dim: 3, lower bound: -0.1234139, upper bound: 0.1234147

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0302720, 1.6531119, 0.0302720, 1.6531119, -1.2185150, 1.2184868
1: -3.3763952, -0.8267784, -3.3763952, -0.8267784, -1.6202102, 1.6201664
2: -0.8847626, -0.3119067, -0.8847626, -0.3119067, -0.2177581, 0.2177579
3: -0.1364306, 0.2343346, -0.1364306, 0.2343346, -0.1792384, 0.1792383
4: -2.7811904, -2.2556832, -2.7811904, -2.2556832, -0.2461713, 0.2461712
5: -0.9392171, -0.5457087, -0.9392171, -0.5457087, -0.1986741, 0.1986714
6: -0.5227640, 0.5105959, -0.5227640, 0.5105959, -0.1530267, 0.1530260
7: -1.0645843, 0.0856791, -1.0645843, 0.0856791, -0.5892586, 0.5892816
8: -4.9083729, -3.1190684, -4.9083729, -3.1190684, -0.9091890, 0.9091079
9: -2.8582137, -1.3061166, -2.8582137, -1.3061166, -0.7476203, 0.7476068

Time for backsubstitution: 5.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 3009
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2480
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 572
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 2588

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 867

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1234238, upper bound: 0.1234232
time: 155.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1234234, upper bound: 0.1234248
time: 82.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0302720, 1.6531119, 0.0302720, 1.6531119, -1.2185150, 1.2184868
1: -3.3763952, -0.8267784, -3.3763952, -0.8267784, -1.6202102, 1.6201664
2: -0.8847626, -0.3119067, -0.8847626, -0.3119067, -0.2177581, 0.2177579
3: -0.1364306, 0.2343346, -0.1364306, 0.2343346, -0.1792384, 0.1792383
4: -2.7811904, -2.2556832, -2.7811904, -2.2556832, -0.2461713, 0.2461712
5: -0.9392171, -0.5457087, -0.9392171, -0.5457087, -0.1986741, 0.1986714
6: -0.5227640, 0.5105959, -0.5227640, 0.5105959, -0.1530267, 0.1530260
7: -1.0645843, 0.0856791, -1.0645843, 0.0856791, -0.5892586, 0.5892816
8: -4.9083729, -3.1190684, -4.9083729, -3.1190684, -0.9091890, 0.9091079
9: -2.8582137, -1.3061166, -2.8582137, -1.3061166, -0.7476203, 0.7476068

Time for backsubstitution: 6.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 3009
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 572
type: DSZ, layer: 1, pos: 2480
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2189

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2376

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233547, upper bound: 0.1234210
time: 15.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1234186, upper bound: 0.1233552
time: 146.10 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0302720, 1.6531119, 0.0302720, 1.6531119, -1.2185040, 1.2185067
1: -3.3763952, -0.8267784, -3.3763952, -0.8267784, -1.6201565, 1.6202153
2: -0.8847626, -0.3119067, -0.8847626, -0.3119067, -0.2177526, 0.2177669
3: -0.1364306, 0.2343346, -0.1364306, 0.2343346, -0.1792378, 0.1792373
4: -2.7811904, -2.2556832, -2.7811904, -2.2556832, -0.2461839, 0.2461655
5: -0.9392171, -0.5457087, -0.9392171, -0.5457087, -0.1986756, 0.1986702
6: -0.5227640, 0.5105959, -0.5227640, 0.5105959, -0.1530258, 0.1530199
7: -1.0645843, 0.0856791, -1.0645843, 0.0856791, -0.5892622, 0.5892788
8: -4.9083729, -3.1190684, -4.9083729, -3.1190684, -0.9091749, 0.9091129
9: -2.8582137, -1.3061166, -2.8582137, -1.3061166, -0.7475996, 0.7476283

Time for backsubstitution: 5.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2480
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 572
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 3009
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2222

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233817, upper bound: 0.1233736
time: 339.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233741, upper bound: 0.1233789
time: 129.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0302720, 1.6531119, 0.0302720, 1.6531119, -1.2185150, 1.2184757
1: -3.3763952, -0.8267784, -3.3763952, -0.8267784, -1.6202102, 1.6201125
2: -0.8847626, -0.3119067, -0.8847626, -0.3119067, -0.2177581, 0.2177524
3: -0.1364306, 0.2343346, -0.1364306, 0.2343346, -0.1792384, 0.1792376
4: -2.7811904, -2.2556832, -2.7811904, -2.2556832, -0.2461655, 0.2461712
5: -0.9392171, -0.5457087, -0.9392171, -0.5457087, -0.1986729, 0.1986714
6: -0.5227640, 0.5105959, -0.5227640, 0.5105959, -0.1530267, 0.1530251
7: -1.0645843, 0.0856791, -1.0645843, 0.0856791, -0.5892558, 0.5892816
8: -4.9083729, -3.1190684, -4.9083729, -3.1190684, -0.9091890, 0.9090940
9: -2.8582137, -1.3061166, -2.8582137, -1.3061166, -0.7476203, 0.7475860

Time for backsubstitution: 6.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 572
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 3009
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2480
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 3446

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2243

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233801, upper bound: 0.1233841
time: 171.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233801, upper bound: 0.1233854
time: 27.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0302720, 1.6531119, 0.0302720, 1.6531119, -1.2179146, 1.2179480
1: -3.3763952, -0.8267784, -3.3763952, -0.8267784, -1.6201570, 1.6202019
2: -0.8847626, -0.3119067, -0.8847626, -0.3119067, -0.2177151, 0.2177233
3: -0.1364306, 0.2343346, -0.1364306, 0.2343346, -0.1792384, 0.1792383
4: -2.7811904, -2.2556832, -2.7811904, -2.2556832, -0.2461432, 0.2461414
5: -0.9392171, -0.5457087, -0.9392171, -0.5457087, -0.1986319, 0.1986327
6: -0.5227640, 0.5105959, -0.5227640, 0.5105959, -0.1529760, 0.1529768
7: -1.0645843, 0.0856791, -1.0645843, 0.0856791, -0.5891690, 0.5891416
8: -4.9083729, -3.1190684, -4.9083729, -3.1190684, -0.9082601, 0.9083487
9: -2.8582137, -1.3061166, -2.8582137, -1.3061166, -0.7475801, 0.7475947

Time for backsubstitution: 6.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3009
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2480
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 572
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2246

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 885

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1234215, upper bound: 0.1234224
time: 42.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1234215, upper bound: 0.1234258
time: 13.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0302720, 1.6531119, 0.0302720, 1.6531119, -1.2179270, 1.2179353
1: -3.3763952, -0.8267784, -3.3763952, -0.8267784, -1.6201572, 1.6202017
2: -0.8847626, -0.3119067, -0.8847626, -0.3119067, -0.2177193, 0.2177191
3: -0.1364306, 0.2343346, -0.1364306, 0.2343346, -0.1792380, 0.1792387
4: -2.7811904, -2.2556832, -2.7811904, -2.2556832, -0.2461417, 0.2461429
5: -0.9392171, -0.5457087, -0.9392171, -0.5457087, -0.1986325, 0.1986321
6: -0.5227640, 0.5105959, -0.5227640, 0.5105959, -0.1529740, 0.1529788
7: -1.0645843, 0.0856791, -1.0645843, 0.0856791, -0.5891795, 0.5891312
8: -4.9083729, -3.1190684, -4.9083729, -3.1190684, -0.9082749, 0.9083338
9: -2.8582137, -1.3061166, -2.8582137, -1.3061166, -0.7475817, 0.7475932

Time for backsubstitution: 6.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2480
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 3009
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 572
type: DSZ, layer: 1, pos: 896

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2212

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1234221, upper bound: 0.1234237
time: 80.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1234201, upper bound: 0.1234262
time: 31.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0302720, 1.6531119, 0.0302720, 1.6531119, -1.2173376, 1.2173663
1: -3.3763952, -0.8267784, -3.3763952, -0.8267784, -1.6159657, 1.6160915
2: -0.8847626, -0.3119067, -0.8847626, -0.3119067, -0.2175608, 0.2175604
3: -0.1364306, 0.2343346, -0.1364306, 0.2343346, -0.1792359, 0.1792357
4: -2.7811904, -2.2556832, -2.7811904, -2.2556832, -0.2461615, 0.2461618
5: -0.9392171, -0.5457087, -0.9392171, -0.5457087, -0.1981810, 0.1981727
6: -0.5227640, 0.5105959, -0.5227640, 0.5105959, -0.1524012, 0.1523959
7: -1.0645843, 0.0856791, -1.0645843, 0.0856791, -0.5880456, 0.5880222
8: -4.9083729, -3.1190684, -4.9083729, -3.1190684, -0.9006407, 0.9007009
9: -2.8582137, -1.3061166, -2.8582137, -1.3061166, -0.7397196, 0.7398009

Time for backsubstitution: 6.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 3009
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2480
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 572
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2378

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 780

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1234231, upper bound: 0.1234136
time: 19.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1234242, upper bound: 0.1234113
time: 140.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0302720, 1.6531119, 0.0302720, 1.6531119, -1.2173306, 1.2173733
1: -3.3763952, -0.8267784, -3.3763952, -0.8267784, -1.6160485, 1.6160090
2: -0.8847626, -0.3119067, -0.8847626, -0.3119067, -0.2175639, 0.2175573
3: -0.1364306, 0.2343346, -0.1364306, 0.2343346, -0.1792358, 0.1792359
4: -2.7811904, -2.2556832, -2.7811904, -2.2556832, -0.2461615, 0.2461619
5: -0.9392171, -0.5457087, -0.9392171, -0.5457087, -0.1981675, 0.1981861
6: -0.5227640, 0.5105959, -0.5227640, 0.5105959, -0.1523974, 0.1523997
7: -1.0645843, 0.0856791, -1.0645843, 0.0856791, -0.5880303, 0.5880374
8: -4.9083729, -3.1190684, -4.9083729, -3.1190684, -0.9006127, 0.9007288
9: -2.8582137, -1.3061166, -2.8582137, -1.3061166, -0.7397867, 0.7397338

Time for backsubstitution: 6.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2480
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 572
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 3009
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2275

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3069

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233593, upper bound: 0.1233931
time: 10.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233812, upper bound: 0.1233646
time: 160.95 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 177.43 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 177.43
Output dim: 3, lower bound: -0.1234238, upper bound: 0.1234232
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 177.43
Output dim: 3, lower bound: -0.1234234, upper bound: 0.1234248
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 177.43
Output dim: 3, lower bound: -0.1233547, upper bound: 0.1234210
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 177.43
Output dim: 3, lower bound: -0.1234186, upper bound: 0.1233552
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 177.43
Output dim: 3, lower bound: -0.1233817, upper bound: 0.1233736
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 177.43
Output dim: 3, lower bound: -0.1233741, upper bound: 0.1233789
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 177.43
Output dim: 3, lower bound: -0.1233801, upper bound: 0.1233841
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 177.43
Output dim: 3, lower bound: -0.1233801, upper bound: 0.1233854
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 177.43
Output dim: 3, lower bound: -0.1234215, upper bound: 0.1234224
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 177.43
Output dim: 3, lower bound: -0.1234215, upper bound: 0.1234258
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 177.43
Output dim: 3, lower bound: -0.1234221, upper bound: 0.1234237
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 177.43
Output dim: 3, lower bound: -0.1234201, upper bound: 0.1234262
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 177.43
Output dim: 3, lower bound: -0.1234231, upper bound: 0.1234136
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 177.43
Output dim: 3, lower bound: -0.1234242, upper bound: 0.1234113
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 177.43
Output dim: 3, lower bound: -0.1233593, upper bound: 0.1233931
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 177.43
Output dim: 3, lower bound: -0.1233812, upper bound: 0.1233646

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0302720, 1.6531119, 0.0302720, 1.6531119, -1.2183902, 1.2183318
1: -3.3763952, -0.8267784, -3.3763952, -0.8267784, -1.6199437, 1.6198484
2: -0.8847626, -0.3119067, -0.8847626, -0.3119067, -0.2177530, 0.2177515
3: -0.1364306, 0.2343346, -0.1364306, 0.2343346, -0.1792230, 0.1792214
4: -2.7811904, -2.2556832, -2.7811904, -2.2556832, -0.2461695, 0.2461694
5: -0.9392171, -0.5457087, -0.9392171, -0.5457087, -0.1986396, 0.1986402
6: -0.5227640, 0.5105959, -0.5227640, 0.5105959, -0.1530140, 0.1530136
7: -1.0645843, 0.0856791, -1.0645843, 0.0856791, -0.5891382, 0.5891790
8: -4.9083729, -3.1190684, -4.9083729, -3.1190684, -0.9086913, 0.9084811
9: -2.8582137, -1.3061166, -2.8582137, -1.3061166, -0.7472677, 0.7471468

Time for backsubstitution: 6.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 3009
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 572
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2480
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2995

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 724

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1234212, upper bound: 0.1234207
time: 187.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1234199, upper bound: 0.1234264
time: 11.33 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0302720, 1.6531119, 0.0302720, 1.6531119, -1.2183602, 1.2183621
1: -3.3763952, -0.8267784, -3.3763952, -0.8267784, -1.6198922, 1.6198996
2: -0.8847626, -0.3119067, -0.8847626, -0.3119067, -0.2177518, 0.2177528
3: -0.1364306, 0.2343346, -0.1364306, 0.2343346, -0.1792215, 0.1792228
4: -2.7811904, -2.2556832, -2.7811904, -2.2556832, -0.2461694, 0.2461694
5: -0.9392171, -0.5457087, -0.9392171, -0.5457087, -0.1986429, 0.1986370
6: -0.5227640, 0.5105959, -0.5227640, 0.5105959, -0.1530143, 0.1530134
7: -1.0645843, 0.0856791, -1.0645843, 0.0856791, -0.5891558, 0.5891613
8: -4.9083729, -3.1190684, -4.9083729, -3.1190684, -0.9085621, 0.9086103
9: -2.8582137, -1.3061166, -2.8582137, -1.3061166, -0.7471604, 0.7472541

Time for backsubstitution: 5.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 572
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2480
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 3009
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 712

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 684

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1234236, upper bound: 0.1234256
time: 15.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1234228, upper bound: 0.1234258
time: 116.97 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0302720, 1.6531119, 0.0302720, 1.6531119, -1.2167785, 1.2166692
1: -3.3763952, -0.8267784, -3.3763952, -0.8267784, -1.6095259, 1.6088190
2: -0.8847626, -0.3119067, -0.8847626, -0.3119067, -0.2167652, 0.2167402
3: -0.1364306, 0.2343346, -0.1364306, 0.2343346, -0.1790269, 0.1790344
4: -2.7811904, -2.2556832, -2.7811904, -2.2556832, -0.2458584, 0.2458691
5: -0.9392171, -0.5457087, -0.9392171, -0.5457087, -0.1984947, 0.1984963
6: -0.5227640, 0.5105959, -0.5227640, 0.5105959, -0.1501711, 0.1503024
7: -1.0645843, 0.0856791, -1.0645843, 0.0856791, -0.5892469, 0.5892712
8: -4.9083729, -3.1190684, -4.9083729, -3.1190684, -0.9067917, 0.9066272
9: -2.8582137, -1.3061166, -2.8582137, -1.3061166, -0.7444448, 0.7442287

Time for backsubstitution: 5.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 572
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2480
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 3009
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2229

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2354

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233525, upper bound: 0.1234190
time: 41.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233525, upper bound: 0.1234223
time: 11.95 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0302720, 1.6531119, 0.0302720, 1.6531119, -1.2166975, 1.2167503
1: -3.3763952, -0.8267784, -3.3763952, -0.8267784, -1.6088629, 1.6094820
2: -0.8847626, -0.3119067, -0.8847626, -0.3119067, -0.2167404, 0.2167649
3: -0.1364306, 0.2343346, -0.1364306, 0.2343346, -0.1790345, 0.1790268
4: -2.7811904, -2.2556832, -2.7811904, -2.2556832, -0.2458692, 0.2458584
5: -0.9392171, -0.5457087, -0.9392171, -0.5457087, -0.1984991, 0.1984920
6: -0.5227640, 0.5105959, -0.5227640, 0.5105959, -0.1503031, 0.1501704
7: -1.0645843, 0.0856791, -1.0645843, 0.0856791, -0.5892483, 0.5892701
8: -4.9083729, -3.1190684, -4.9083729, -3.1190684, -0.9067081, 0.9067107
9: -2.8582137, -1.3061166, -2.8582137, -1.3061166, -0.7442424, 0.7444313

Time for backsubstitution: 5.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 572
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2480
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 3009
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2988

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2298

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1234063, upper bound: 0.1233516
time: 58.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1234151, upper bound: 0.1233451
time: 72.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0302720, 1.6531119, 0.0302720, 1.6531119, -1.2184948, 1.2184973
1: -3.3763952, -0.8267784, -3.3763952, -0.8267784, -1.6196043, 1.6196574
2: -0.8847626, -0.3119067, -0.8847626, -0.3119067, -0.2175384, 0.2175533
3: -0.1364306, 0.2343346, -0.1364306, 0.2343346, -0.1790075, 0.1790064
4: -2.7811904, -2.2556832, -2.7811904, -2.2556832, -0.2461402, 0.2461219
5: -0.9392171, -0.5457087, -0.9392171, -0.5457087, -0.1986696, 0.1986620
6: -0.5227640, 0.5105959, -0.5227640, 0.5105959, -0.1522582, 0.1522507
7: -1.0645843, 0.0856791, -1.0645843, 0.0856791, -0.5892766, 0.5892931
8: -4.9083729, -3.1190684, -4.9083729, -3.1190684, -0.9091752, 0.9091141
9: -2.8582137, -1.3061166, -2.8582137, -1.3061166, -0.7467918, 0.7468155

Time for backsubstitution: 5.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2480
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 572
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 3009
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2241

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2036

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233771, upper bound: 0.1233734
time: 81.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233816, upper bound: 0.1233726
time: 145.97 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0302720, 1.6531119, 0.0302720, 1.6531119, -1.2184948, 1.2184975
1: -3.3763952, -0.8267784, -3.3763952, -0.8267784, -1.6195984, 1.6196634
2: -0.8847626, -0.3119067, -0.8847626, -0.3119067, -0.2175390, 0.2175527
3: -0.1364306, 0.2343346, -0.1364306, 0.2343346, -0.1790069, 0.1790071
4: -2.7811904, -2.2556832, -2.7811904, -2.2556832, -0.2461402, 0.2461218
5: -0.9392171, -0.5457087, -0.9392171, -0.5457087, -0.1986674, 0.1986642
6: -0.5227640, 0.5105959, -0.5227640, 0.5105959, -0.1522566, 0.1522523
7: -1.0645843, 0.0856791, -1.0645843, 0.0856791, -0.5892766, 0.5892931
8: -4.9083729, -3.1190684, -4.9083729, -3.1190684, -0.9091761, 0.9091130
9: -2.8582137, -1.3061166, -2.8582137, -1.3061166, -0.7467868, 0.7468206

Time for backsubstitution: 5.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 3009
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 572
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2480
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 692

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 81

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233344, upper bound: 0.1233665
time: 96.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233633, upper bound: 0.1233404
time: 44.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0302720, 1.6531119, 0.0302720, 1.6531119, -1.2185150, 1.2184757
1: -3.3763952, -0.8267784, -3.3763952, -0.8267784, -1.6202102, 1.6201125
2: -0.8847626, -0.3119067, -0.8847626, -0.3119067, -0.2177581, 0.2177524
3: -0.1364306, 0.2343346, -0.1364306, 0.2343346, -0.1792384, 0.1792376
4: -2.7811904, -2.2556832, -2.7811904, -2.2556832, -0.2461655, 0.2461712
5: -0.9392171, -0.5457087, -0.9392171, -0.5457087, -0.1986729, 0.1986714
6: -0.5227640, 0.5105959, -0.5227640, 0.5105959, -0.1530267, 0.1530251
7: -1.0645843, 0.0856791, -1.0645843, 0.0856791, -0.5892558, 0.5892816
8: -4.9083729, -3.1190684, -4.9083729, -3.1190684, -0.9091890, 0.9090940
9: -2.8582137, -1.3061166, -2.8582137, -1.3061166, -0.7476203, 0.7475860

Time for backsubstitution: 5.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2480
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 3009
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 572
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2948

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3001

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233745, upper bound: 0.1233811
time: 224.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233762, upper bound: 0.1233815
time: 88.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0302720, 1.6531119, 0.0302720, 1.6531119, -1.2185150, 1.2184757
1: -3.3763952, -0.8267784, -3.3763952, -0.8267784, -1.6202102, 1.6201125
2: -0.8847626, -0.3119067, -0.8847626, -0.3119067, -0.2177581, 0.2177524
3: -0.1364306, 0.2343346, -0.1364306, 0.2343346, -0.1792384, 0.1792376
4: -2.7811904, -2.2556832, -2.7811904, -2.2556832, -0.2461655, 0.2461712
5: -0.9392171, -0.5457087, -0.9392171, -0.5457087, -0.1986729, 0.1986714
6: -0.5227640, 0.5105959, -0.5227640, 0.5105959, -0.1530267, 0.1530251
7: -1.0645843, 0.0856791, -1.0645843, 0.0856791, -0.5892558, 0.5892816
8: -4.9083729, -3.1190684, -4.9083729, -3.1190684, -0.9091890, 0.9090940
9: -2.8582137, -1.3061166, -2.8582137, -1.3061166, -0.7476203, 0.7475860

Time for backsubstitution: 5.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 3009
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2480
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 572
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2111

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2273

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233787, upper bound: 0.1233897
time: 6.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233791, upper bound: 0.1233846
time: 101.75 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 114.20 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 114.20
Output dim: 3, lower bound: -0.1234212, upper bound: 0.1234207
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 114.20
Output dim: 3, lower bound: -0.1234199, upper bound: 0.1234264
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 114.20
Output dim: 3, lower bound: -0.1234236, upper bound: 0.1234256
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 114.20
Output dim: 3, lower bound: -0.1234228, upper bound: 0.1234258
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 114.20
Output dim: 3, lower bound: -0.1233525, upper bound: 0.1234190
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 114.20
Output dim: 3, lower bound: -0.1233525, upper bound: 0.1234223
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 114.20
Output dim: 3, lower bound: -0.1234063, upper bound: 0.1233516
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 114.20
Output dim: 3, lower bound: -0.1234151, upper bound: 0.1233451
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 114.20
Output dim: 3, lower bound: -0.1233771, upper bound: 0.1233734
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 114.20
Output dim: 3, lower bound: -0.1233816, upper bound: 0.1233726
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 114.20
Output dim: 3, lower bound: -0.1233344, upper bound: 0.1233665
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 114.20
Output dim: 3, lower bound: -0.1233633, upper bound: 0.1233404
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 114.20
Output dim: 3, lower bound: -0.1233745, upper bound: 0.1233811
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 114.20
Output dim: 3, lower bound: -0.1233762, upper bound: 0.1233815
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 114.20
Output dim: 3, lower bound: -0.1233787, upper bound: 0.1233897
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 114.20
Output dim: 3, lower bound: -0.1233791, upper bound: 0.1233846
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 114.20
Output dim: 3, lower bound: -0.1234215, upper bound: 0.1234224
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 114.20
Output dim: 3, lower bound: -0.1234215, upper bound: 0.1234258
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 114.20
Output dim: 3, lower bound: -0.1234221, upper bound: 0.1234237
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 114.20
Output dim: 3, lower bound: -0.1234201, upper bound: 0.1234262
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 114.20
Output dim: 3, lower bound: -0.1234231, upper bound: 0.1234136
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 114.20
Output dim: 3, lower bound: -0.1234242, upper bound: 0.1234113
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 114.20
Output dim: 3, lower bound: -0.1233593, upper bound: 0.1233931
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 114.20
Output dim: 3, lower bound: -0.1233812, upper bound: 0.1233646

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 71.27 + 3552.60 = 3623.87 seconds
