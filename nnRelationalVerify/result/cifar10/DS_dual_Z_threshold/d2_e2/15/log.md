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
execution time: IAR + RelationalAnalysis = 8.18 + 63.20 = 71.38 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.1234413, upper bound: 0.1234429

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 572
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 3009
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2480
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3246

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2390

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233797, upper bound: 0.1233810
time: 264.06 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1234080, upper bound: 0.1233850
time: 9.23 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 273.37 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 273.37
Output dim: 3, lower bound: -0.1233797, upper bound: 0.1233810
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 273.37
Output dim: 3, lower bound: -0.1234080, upper bound: 0.1233850

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 0.0302720, 1.6531119, 0.0302720, 1.6531119, -1.2181797, 1.2181673
1: -3.3763952, -0.8267784, -3.3763952, -0.8267784, -1.6178942, 1.6176780
2: -0.8847626, -0.3119067, -0.8847626, -0.3119067, -0.2175667, 0.2175635
3: -0.1364306, 0.2343346, -0.1364306, 0.2343346, -0.1792181, 0.1792184
4: -2.7811904, -2.2556832, -2.7811904, -2.2556832, -0.2461450, 0.2461475
5: -0.9392171, -0.5457087, -0.9392171, -0.5457087, -0.1986145, 0.1986188
6: -0.5227640, 0.5105959, -0.5227640, 0.5105959, -0.1529326, 0.1529366
7: -1.0645843, 0.0856791, -1.0645843, 0.0856791, -0.5891575, 0.5891755
8: -4.9083729, -3.1190684, -4.9083729, -3.1190684, -0.9084768, 0.9084039
9: -2.8582137, -1.3061166, -2.8582137, -1.3061166, -0.7460794, 0.7459285

Time for backsubstitution: 5.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 572
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 3009
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2480
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3246

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 155

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233335, upper bound: 0.1234104
time: 14.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233793, upper bound: 0.1233654
time: 11.62 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 0.0302720, 1.6531119, 0.0302720, 1.6531119, -1.2181673, 1.2181797
1: -3.3763952, -0.8267784, -3.3763952, -0.8267784, -1.6176779, 1.6178941
2: -0.8847626, -0.3119067, -0.8847626, -0.3119067, -0.2175635, 0.2175667
3: -0.1364306, 0.2343346, -0.1364306, 0.2343346, -0.1792184, 0.1792181
4: -2.7811904, -2.2556832, -2.7811904, -2.2556832, -0.2461475, 0.2461450
5: -0.9392171, -0.5457087, -0.9392171, -0.5457087, -0.1986188, 0.1986145
6: -0.5227640, 0.5105959, -0.5227640, 0.5105959, -0.1529366, 0.1529326
7: -1.0645843, 0.0856791, -1.0645843, 0.0856791, -0.5891753, 0.5891575
8: -4.9083729, -3.1190684, -4.9083729, -3.1190684, -0.9084039, 0.9084768
9: -2.8582137, -1.3061166, -2.8582137, -1.3061166, -0.7459285, 0.7460794

Time for backsubstitution: 5.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 572
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 3009
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2480
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3246

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 155

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233622, upper bound: 0.1233799
time: 120.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1234063, upper bound: 0.1233380
time: 16.49 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 142.89 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 142.89
Output dim: 3, lower bound: -0.1233335, upper bound: 0.1234104
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 142.89
Output dim: 3, lower bound: -0.1233793, upper bound: 0.1233654
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 142.89
Output dim: 3, lower bound: -0.1233622, upper bound: 0.1233799
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 142.89
Output dim: 3, lower bound: -0.1234063, upper bound: 0.1233380

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0302720, 1.6531119, 0.0302720, 1.6531119, -1.2179991, 1.2179819
1: -3.3763952, -0.8267784, -3.3763952, -0.8267784, -1.6170750, 1.6168377
2: -0.8847626, -0.3119067, -0.8847626, -0.3119067, -0.2174641, 0.2174614
3: -0.1364306, 0.2343346, -0.1364306, 0.2343346, -0.1792393, 0.1792398
4: -2.7811904, -2.2556832, -2.7811904, -2.2556832, -0.2461381, 0.2461411
5: -0.9392171, -0.5457087, -0.9392171, -0.5457087, -0.1987033, 0.1987075
6: -0.5227640, 0.5105959, -0.5227640, 0.5105959, -0.1529509, 0.1529556
7: -1.0645843, 0.0856791, -1.0645843, 0.0856791, -0.5891572, 0.5891751
8: -4.9083729, -3.1190684, -4.9083729, -3.1190684, -0.9079880, 0.9079084
9: -2.8582137, -1.3061166, -2.8582137, -1.3061166, -0.7456868, 0.7455375

Time for backsubstitution: 5.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 572
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 3009
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2480
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3246

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2164

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1231879, upper bound: 0.1234001
time: 8.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233223, upper bound: 0.1232724
time: 79.11 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0302720, 1.6531119, 0.0302720, 1.6531119, -1.2179943, 1.2179867
1: -3.3763952, -0.8267784, -3.3763952, -0.8267784, -1.6170540, 1.6168587
2: -0.8847626, -0.3119067, -0.8847626, -0.3119067, -0.2174646, 0.2174610
3: -0.1364306, 0.2343346, -0.1364306, 0.2343346, -0.1792395, 0.1792395
4: -2.7811904, -2.2556832, -2.7811904, -2.2556832, -0.2461386, 0.2461406
5: -0.9392171, -0.5457087, -0.9392171, -0.5457087, -0.1987032, 0.1987075
6: -0.5227640, 0.5105959, -0.5227640, 0.5105959, -0.1529516, 0.1529550
7: -1.0645843, 0.0856791, -1.0645843, 0.0856791, -0.5891572, 0.5891751
8: -4.9083729, -3.1190684, -4.9083729, -3.1190684, -0.9079815, 0.9079149
9: -2.8582137, -1.3061166, -2.8582137, -1.3061166, -0.7456884, 0.7455358

Time for backsubstitution: 6.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 572
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 3009
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2480
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3246

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2164

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1232330, upper bound: 0.1233552
time: 8.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233700, upper bound: 0.1232202
time: 49.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0302720, 1.6531119, 0.0302720, 1.6531119, -1.2179867, 1.2179943
1: -3.3763952, -0.8267784, -3.3763952, -0.8267784, -1.6168587, 1.6170540
2: -0.8847626, -0.3119067, -0.8847626, -0.3119067, -0.2174610, 0.2174646
3: -0.1364306, 0.2343346, -0.1364306, 0.2343346, -0.1792396, 0.1792395
4: -2.7811904, -2.2556832, -2.7811904, -2.2556832, -0.2461406, 0.2461386
5: -0.9392171, -0.5457087, -0.9392171, -0.5457087, -0.1987076, 0.1987032
6: -0.5227640, 0.5105959, -0.5227640, 0.5105959, -0.1529550, 0.1529516
7: -1.0645843, 0.0856791, -1.0645843, 0.0856791, -0.5891751, 0.5891571
8: -4.9083729, -3.1190684, -4.9083729, -3.1190684, -0.9079150, 0.9079815
9: -2.8582137, -1.3061166, -2.8582137, -1.3061166, -0.7455356, 0.7456884

Time for backsubstitution: 6.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 572
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 3009
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2480
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3246

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2164

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1232243, upper bound: 0.1233737
time: 18.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233512, upper bound: 0.1231914
time: 76.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0302720, 1.6531119, 0.0302720, 1.6531119, -1.2179819, 1.2179991
1: -3.3763952, -0.8267784, -3.3763952, -0.8267784, -1.6168377, 1.6170750
2: -0.8847626, -0.3119067, -0.8847626, -0.3119067, -0.2174615, 0.2174641
3: -0.1364306, 0.2343346, -0.1364306, 0.2343346, -0.1792398, 0.1792393
4: -2.7811904, -2.2556832, -2.7811904, -2.2556832, -0.2461411, 0.2461381
5: -0.9392171, -0.5457087, -0.9392171, -0.5457087, -0.1987075, 0.1987033
6: -0.5227640, 0.5105959, -0.5227640, 0.5105959, -0.1529556, 0.1529509
7: -1.0645843, 0.0856791, -1.0645843, 0.0856791, -0.5891751, 0.5891571
8: -4.9083729, -3.1190684, -4.9083729, -3.1190684, -0.9079086, 0.9079880
9: -2.8582137, -1.3061166, -2.8582137, -1.3061166, -0.7455375, 0.7456868

Time for backsubstitution: 6.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 572
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 3009
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2480
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3246

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2164

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1232687, upper bound: 0.1233281
time: 12.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1233949, upper bound: 0.1231941
time: 8.61 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 27.72 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 27.72
Output dim: 3, lower bound: -0.1231879, upper bound: 0.1234001
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 27.72
Output dim: 3, lower bound: -0.1233223, upper bound: 0.1232724
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 27.72
Output dim: 3, lower bound: -0.1232330, upper bound: 0.1233552
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 27.72
Output dim: 3, lower bound: -0.1233700, upper bound: 0.1232202
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 27.72
Output dim: 3, lower bound: -0.1232243, upper bound: 0.1233737
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 27.72
Output dim: 3, lower bound: -0.1233512, upper bound: 0.1231914
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 27.72
Output dim: 3, lower bound: -0.1232687, upper bound: 0.1233281
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 27.72
Output dim: 3, lower bound: -0.1233949, upper bound: 0.1231941

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0302720, 1.6531119, 0.0302720, 1.6531119, -1.2179191, 1.2178417
1: -3.3763952, -0.8267784, -3.3763952, -0.8267784, -1.6167278, 1.6162150
2: -0.8847626, -0.3119067, -0.8847626, -0.3119067, -0.2174172, 0.2174131
3: -0.1364306, 0.2343346, -0.1364306, 0.2343346, -0.1792271, 0.1792278
4: -2.7811904, -2.2556832, -2.7811904, -2.2556832, -0.2461346, 0.2461375
5: -0.9392171, -0.5457087, -0.9392171, -0.5457087, -0.1986830, 0.1986976
6: -0.5227640, 0.5105959, -0.5227640, 0.5105959, -0.1528999, 0.1529153
7: -1.0645843, 0.0856791, -1.0645843, 0.0856791, -0.5891308, 0.5891535
8: -4.9083729, -3.1190684, -4.9083729, -3.1190684, -0.9078157, 0.9075555
9: -2.8582137, -1.3061166, -2.8582137, -1.3061166, -0.7451524, 0.7449944

Time for backsubstitution: 6.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 572
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 3009
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2480
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3246

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 139

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1230765, upper bound: 0.1232832
time: 9.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1230825, upper bound: 0.1232802
time: 7.95 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0302720, 1.6531119, 0.0302720, 1.6531119, -1.2178588, 1.2179028
1: -3.3763952, -0.8267784, -3.3763952, -0.8267784, -1.6164520, 1.6164944
2: -0.8847626, -0.3119067, -0.8847626, -0.3119067, -0.2174158, 0.2174147
3: -0.1364306, 0.2343346, -0.1364306, 0.2343346, -0.1792275, 0.1792276
4: -2.7811904, -2.2556832, -2.7811904, -2.2556832, -0.2461345, 0.2461376
5: -0.9392171, -0.5457087, -0.9392171, -0.5457087, -0.1986936, 0.1986872
6: -0.5227640, 0.5105959, -0.5227640, 0.5105959, -0.1529115, 0.1529045
7: -1.0645843, 0.0856791, -1.0645843, 0.0856791, -0.5891358, 0.5891488
8: -4.9083729, -3.1190684, -4.9083729, -3.1190684, -0.9076350, 0.9077383
9: -2.8582137, -1.3061166, -2.8582137, -1.3061166, -0.7451438, 0.7450068

Time for backsubstitution: 6.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 572
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 3009
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2480
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3246

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 139

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1232031, upper bound: 0.1231645
time: 50.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1232055, upper bound: 0.1231547
time: 78.09 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0302720, 1.6531119, 0.0302720, 1.6531119, -1.2179143, 1.2178465
1: -3.3763952, -0.8267784, -3.3763952, -0.8267784, -1.6167066, 1.6162360
2: -0.8847626, -0.3119067, -0.8847626, -0.3119067, -0.2174176, 0.2174127
3: -0.1364306, 0.2343346, -0.1364306, 0.2343346, -0.1792273, 0.1792276
4: -2.7811904, -2.2556832, -2.7811904, -2.2556832, -0.2461351, 0.2461370
5: -0.9392171, -0.5457087, -0.9392171, -0.5457087, -0.1986830, 0.1986976
6: -0.5227640, 0.5105959, -0.5227640, 0.5105959, -0.1529005, 0.1529147
7: -1.0645843, 0.0856791, -1.0645843, 0.0856791, -0.5891308, 0.5891535
8: -4.9083729, -3.1190684, -4.9083729, -3.1190684, -0.9078093, 0.9075619
9: -2.8582137, -1.3061166, -2.8582137, -1.3061166, -0.7451540, 0.7449927

Time for backsubstitution: 6.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 572
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 3009
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2480
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3246

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 139

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1231224, upper bound: 0.1232379
time: 7.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1231299, upper bound: 0.1232325
time: 25.86 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0302720, 1.6531119, 0.0302720, 1.6531119, -1.2178540, 1.2179075
1: -3.3763952, -0.8267784, -3.3763952, -0.8267784, -1.6164310, 1.6165154
2: -0.8847626, -0.3119067, -0.8847626, -0.3119067, -0.2174162, 0.2174143
3: -0.1364306, 0.2343346, -0.1364306, 0.2343346, -0.1792278, 0.1792274
4: -2.7811904, -2.2556832, -2.7811904, -2.2556832, -0.2461350, 0.2461371
5: -0.9392171, -0.5457087, -0.9392171, -0.5457087, -0.1986936, 0.1986873
6: -0.5227640, 0.5105959, -0.5227640, 0.5105959, -0.1529121, 0.1529039
7: -1.0645843, 0.0856791, -1.0645843, 0.0856791, -0.5891358, 0.5891488
8: -4.9083729, -3.1190684, -4.9083729, -3.1190684, -0.9076285, 0.9077449
9: -2.8582137, -1.3061166, -2.8582137, -1.3061166, -0.7451454, 0.7450051

Time for backsubstitution: 6.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 572
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 3009
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2480
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3246

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 139

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1232480, upper bound: 0.1231185
time: 38.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1232530, upper bound: 0.1231125
time: 100.10 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.0302720, 1.6531119, 0.0302720, 1.6531119, -1.2179077, 1.2178541
1: -3.3763952, -0.8267784, -3.3763952, -0.8267784, -1.6165154, 1.6164310
2: -0.8847626, -0.3119067, -0.8847626, -0.3119067, -0.2174143, 0.2174163
3: -0.1364306, 0.2343346, -0.1364306, 0.2343346, -0.1792274, 0.1792277
4: -2.7811904, -2.2556832, -2.7811904, -2.2556832, -0.2461371, 0.2461350
5: -0.9392171, -0.5457087, -0.9392171, -0.5457087, -0.1986873, 0.1986936
6: -0.5227640, 0.5105959, -0.5227640, 0.5105959, -0.1529039, 0.1529121
7: -1.0645843, 0.0856791, -1.0645843, 0.0856791, -0.5891486, 0.5891359
8: -4.9083729, -3.1190684, -4.9083729, -3.1190684, -0.9077449, 0.9076284
9: -2.8582137, -1.3061166, -2.8582137, -1.3061166, -0.7450050, 0.7451454

Time for backsubstitution: 6.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 572
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 3009
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2480
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3246

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 139

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1231096, upper bound: 0.1232565
time: 7.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1231168, upper bound: 0.1232033
time: 230.94 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.0302720, 1.6531119, 0.0302720, 1.6531119, -1.2178464, 1.2179145
1: -3.3763952, -0.8267784, -3.3763952, -0.8267784, -1.6162360, 1.6167066
2: -0.8847626, -0.3119067, -0.8847626, -0.3119067, -0.2174126, 0.2174177
3: -0.1364306, 0.2343346, -0.1364306, 0.2343346, -0.1792276, 0.1792273
4: -2.7811904, -2.2556832, -2.7811904, -2.2556832, -0.2461370, 0.2461350
5: -0.9392171, -0.5457087, -0.9392171, -0.5457087, -0.1986976, 0.1986830
6: -0.5227640, 0.5105959, -0.5227640, 0.5105959, -0.1529147, 0.1529005
7: -1.0645843, 0.0856791, -1.0645843, 0.0856791, -0.5891534, 0.5891308
8: -4.9083729, -3.1190684, -4.9083729, -3.1190684, -0.9075620, 0.9078091
9: -2.8582137, -1.3061166, -2.8582137, -1.3061166, -0.7449926, 0.7451540

Time for backsubstitution: 6.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 572
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 3009
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2480
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3246

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 139

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1232298, upper bound: 0.1231306
time: 9.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1232326, upper bound: 0.1231187
time: 144.80 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.0302720, 1.6531119, 0.0302720, 1.6531119, -1.2179029, 1.2178589
1: -3.3763952, -0.8267784, -3.3763952, -0.8267784, -1.6164944, 1.6164520
2: -0.8847626, -0.3119067, -0.8847626, -0.3119067, -0.2174147, 0.2174158
3: -0.1364306, 0.2343346, -0.1364306, 0.2343346, -0.1792276, 0.1792275
4: -2.7811904, -2.2556832, -2.7811904, -2.2556832, -0.2461376, 0.2461345
5: -0.9392171, -0.5457087, -0.9392171, -0.5457087, -0.1986873, 0.1986936
6: -0.5227640, 0.5105959, -0.5227640, 0.5105959, -0.1529045, 0.1529115
7: -1.0645843, 0.0856791, -1.0645843, 0.0856791, -0.5891486, 0.5891359
8: -4.9083729, -3.1190684, -4.9083729, -3.1190684, -0.9077384, 0.9076349
9: -2.8582137, -1.3061166, -2.8582137, -1.3061166, -0.7450069, 0.7451437

Time for backsubstitution: 6.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 572
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 3009
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2480
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3246

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 139

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1231568, upper bound: 0.1232066
time: 155.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1231612, upper bound: 0.1232046
time: 100.99 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.0302720, 1.6531119, 0.0302720, 1.6531119, -1.2178416, 1.2179190
1: -3.3763952, -0.8267784, -3.3763952, -0.8267784, -1.6162150, 1.6167276
2: -0.8847626, -0.3119067, -0.8847626, -0.3119067, -0.2174131, 0.2174172
3: -0.1364306, 0.2343346, -0.1364306, 0.2343346, -0.1792278, 0.1792271
4: -2.7811904, -2.2556832, -2.7811904, -2.2556832, -0.2461375, 0.2461345
5: -0.9392171, -0.5457087, -0.9392171, -0.5457087, -0.1986976, 0.1986830
6: -0.5227640, 0.5105959, -0.5227640, 0.5105959, -0.1529153, 0.1528999
7: -1.0645843, 0.0856791, -1.0645843, 0.0856791, -0.5891534, 0.5891308
8: -4.9083729, -3.1190684, -4.9083729, -3.1190684, -0.9075556, 0.9078156
9: -2.8582137, -1.3061166, -2.8582137, -1.3061166, -0.7449945, 0.7451523

Time for backsubstitution: 6.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 549
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 565
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 572
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 3009
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2137
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2294
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2480
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3246

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 139

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1232764, upper bound: 0.1230825
time: 179.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.1232792, upper bound: 0.1230788
time: 33.33 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 219.41 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 219.41
Output dim: 3, lower bound: -0.1230765, upper bound: 0.1232832
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 219.41
Output dim: 3, lower bound: -0.1230825, upper bound: 0.1232802
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 219.41
Output dim: 3, lower bound: -0.1232031, upper bound: 0.1231645
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 219.41
Output dim: 3, lower bound: -0.1232055, upper bound: 0.1231547
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 219.41
Output dim: 3, lower bound: -0.1231224, upper bound: 0.1232379
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 219.41
Output dim: 3, lower bound: -0.1231299, upper bound: 0.1232325
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 219.41
Output dim: 3, lower bound: -0.1232480, upper bound: 0.1231185
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 219.41
Output dim: 3, lower bound: -0.1232530, upper bound: 0.1231125
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 219.41
Output dim: 3, lower bound: -0.1231096, upper bound: 0.1232565
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 219.41
Output dim: 3, lower bound: -0.1231168, upper bound: 0.1232033
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 219.41
Output dim: 3, lower bound: -0.1232298, upper bound: 0.1231306
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 219.41
Output dim: 3, lower bound: -0.1232326, upper bound: 0.1231187
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 219.41
Output dim: 3, lower bound: -0.1231568, upper bound: 0.1232066
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 219.41
Output dim: 3, lower bound: -0.1231612, upper bound: 0.1232046
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 219.41
Output dim: 3, lower bound: -0.1232764, upper bound: 0.1230825
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 219.41
Output dim: 3, lower bound: -0.1232792, upper bound: 0.1230788

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 71.38 + 1966.46 = 2037.84 seconds
