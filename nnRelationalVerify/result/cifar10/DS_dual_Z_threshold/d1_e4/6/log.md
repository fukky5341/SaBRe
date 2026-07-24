## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 6)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.1188709816


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-5.1625791, -3.6230450, -5.1625791, -3.6230450, -0.6350532, 0.6350533)
1: (-3.3887079, -1.6014743, -3.3887079, -1.6014743, -0.9551976, 0.9551976)
2: (-1.8911505, -1.3154993, -1.8911505, -1.3154993, -0.3909123, 0.3909123)
3: (-0.0171260, 0.5609033, -0.0171260, 0.5609033, -0.4710022, 0.4710022)
4: (-2.2231789, -1.5974650, -2.2231789, -1.5974650, -0.2500200, 0.2500200)
5: (-0.6258286, 0.0854708, -0.6258286, 0.0854708, -0.5883187, 0.5883187)
6: (0.1040056, 0.7396585, 0.1040056, 0.7396585, -0.5065912, 0.5065913)
7: (-1.4028728, -0.7634962, -1.4028728, -0.7634962, -0.1603919, 0.1603919)
8: (-4.9364576, -3.5115349, -4.9364576, -3.5115349, -0.5953761, 0.5953760)
9: (-0.6763513, 0.4862144, -0.6763513, 0.4862144, -0.8073617, 0.8073617)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.87 + 91.84 = 99.71 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.1191051, upper bound: 0.1191087

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 3425
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 516
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 55
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 3455
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 291
type: DSZ, layer: 1, pos: 3243
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 319
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 2614
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 3481
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 3424
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3465
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 3454
type: DSZ, layer: 1, pos: 3202
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 290
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 3416
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 492
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 508
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3572
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 433
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 628
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 2338
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2651
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2963
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3188

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 306

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1183170, upper bound: 0.1190996
time: 238.66 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1190970, upper bound: 0.1183187
time: 280.63 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 519.37 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 519.37
Output dim: 6, lower bound: -0.1183170, upper bound: 0.1190996
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 519.37
Output dim: 6, lower bound: -0.1190970, upper bound: 0.1183187

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -5.1625791, -3.6230450, -5.1625791, -3.6230450, -0.6319904, 0.6319024
1: -3.3887079, -1.6014743, -3.3887079, -1.6014743, -0.9551525, 0.9551511
2: -1.8911505, -1.3154993, -1.8911505, -1.3154993, -0.3904555, 0.3904459
3: -0.0171260, 0.5609033, -0.0171260, 0.5609033, -0.4704161, 0.4704335
4: -2.2231789, -1.5974650, -2.2231789, -1.5974650, -0.2435036, 0.2432837
5: -0.6258286, 0.0854708, -0.6258286, 0.0854708, -0.5878765, 0.5878904
6: 0.1040056, 0.7396585, 0.1040056, 0.7396585, -0.4934953, 0.4938251
7: -1.4028728, -0.7634962, -1.4028728, -0.7634962, -0.1589382, 0.1589170
8: -4.9364576, -3.5115349, -4.9364576, -3.5115349, -0.5941911, 0.5941406
9: -0.6763513, 0.4862144, -0.6763513, 0.4862144, -0.8074636, 0.8074589

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 3425
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 516
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 55
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 3455
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 291
type: DSZ, layer: 1, pos: 3243
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 319
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 2614
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 3481
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 3424
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3465
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 3454
type: DSZ, layer: 1, pos: 3202
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 290
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 3416
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 492
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 508
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3572
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 433
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 628
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 2338
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2651
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2963
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3188

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3071

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1182625, upper bound: 0.1190760
time: 35.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1182905, upper bound: 0.1190469
time: 139.79 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -5.1625791, -3.6230450, -5.1625791, -3.6230450, -0.6319025, 0.6319904
1: -3.3887079, -1.6014743, -3.3887079, -1.6014743, -0.9551513, 0.9551523
2: -1.8911505, -1.3154993, -1.8911505, -1.3154993, -0.3904459, 0.3904555
3: -0.0171260, 0.5609033, -0.0171260, 0.5609033, -0.4704335, 0.4704161
4: -2.2231789, -1.5974650, -2.2231789, -1.5974650, -0.2432837, 0.2435036
5: -0.6258286, 0.0854708, -0.6258286, 0.0854708, -0.5878904, 0.5878764
6: 0.1040056, 0.7396585, 0.1040056, 0.7396585, -0.4938252, 0.4934953
7: -1.4028728, -0.7634962, -1.4028728, -0.7634962, -0.1589170, 0.1589382
8: -4.9364576, -3.5115349, -4.9364576, -3.5115349, -0.5941406, 0.5941911
9: -0.6763513, 0.4862144, -0.6763513, 0.4862144, -0.8074591, 0.8074637

Time for backsubstitution: 6.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 3425
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 516
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 55
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 3455
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 291
type: DSZ, layer: 1, pos: 3243
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 319
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 2614
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 3481
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 3424
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3465
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 3454
type: DSZ, layer: 1, pos: 3202
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 290
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 3416
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 492
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 508
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3572
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 433
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 628
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 2338
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2651
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2963
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3188

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3071

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1190430, upper bound: 0.1182950
time: 197.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1190724, upper bound: 0.1182666
time: 139.99 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 343.88 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 343.88
Output dim: 6, lower bound: -0.1182625, upper bound: 0.1190760
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 343.88
Output dim: 6, lower bound: -0.1182905, upper bound: 0.1190469
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 343.88
Output dim: 6, lower bound: -0.1190430, upper bound: 0.1182950
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 343.88
Output dim: 6, lower bound: -0.1190724, upper bound: 0.1182666

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.1625791, -3.6230450, -5.1625791, -3.6230450, -0.6303939, 0.6302237
1: -3.3887079, -1.6014743, -3.3887079, -1.6014743, -0.9530545, 0.9528258
2: -1.8911505, -1.3154993, -1.8911505, -1.3154993, -0.3904008, 0.3903971
3: -0.0171260, 0.5609033, -0.0171260, 0.5609033, -0.4703821, 0.4704003
4: -2.2231789, -1.5974650, -2.2231789, -1.5974650, -0.2434425, 0.2432273
5: -0.6258286, 0.0854708, -0.6258286, 0.0854708, -0.5878016, 0.5878211
6: 0.1040056, 0.7396585, 0.1040056, 0.7396585, -0.4934922, 0.4938223
7: -1.4028728, -0.7634962, -1.4028728, -0.7634962, -0.1587515, 0.1587529
8: -4.9364576, -3.5115349, -4.9364576, -3.5115349, -0.5929610, 0.5928370
9: -0.6763513, 0.4862144, -0.6763513, 0.4862144, -0.8069509, 0.8068907

Time for backsubstitution: 6.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 3425
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 516
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 55
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 3455
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 291
type: DSZ, layer: 1, pos: 3243
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 319
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 2614
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 3481
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 3424
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3465
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 3454
type: DSZ, layer: 1, pos: 3202
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 290
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 3416
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 492
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 508
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3572
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 433
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 628
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 2338
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2651
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2963
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3188

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2396

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1182190, upper bound: 0.1190413
time: 291.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1182295, upper bound: 0.1190309
time: 32.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.1625791, -3.6230450, -5.1625791, -3.6230450, -0.6303116, 0.6303061
1: -3.3887079, -1.6014743, -3.3887079, -1.6014743, -0.9528271, 0.9530532
2: -1.8911505, -1.3154993, -1.8911505, -1.3154993, -0.3904066, 0.3903913
3: -0.0171260, 0.5609033, -0.0171260, 0.5609033, -0.4703829, 0.4703995
4: -2.2231789, -1.5974650, -2.2231789, -1.5974650, -0.2434473, 0.2432225
5: -0.6258286, 0.0854708, -0.6258286, 0.0854708, -0.5878072, 0.5878155
6: 0.1040056, 0.7396585, 0.1040056, 0.7396585, -0.4934925, 0.4938221
7: -1.4028728, -0.7634962, -1.4028728, -0.7634962, -0.1587741, 0.1587303
8: -4.9364576, -3.5115349, -4.9364576, -3.5115349, -0.5928874, 0.5929106
9: -0.6763513, 0.4862144, -0.6763513, 0.4862144, -0.8068954, 0.8069462

Time for backsubstitution: 6.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 796
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2966
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 3425
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 3241
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 516
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 55
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 3455
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 291
type: DSZ, layer: 1, pos: 3243
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 3175
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 319
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 2614
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 3481
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 3424
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3465
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 3454
type: DSZ, layer: 1, pos: 3202
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 290
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2282
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 3416
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 492
type: DSZ, layer: 1, pos: 3095
type: DSZ, layer: 1, pos: 508
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3208
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3572
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 433
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 628
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 2338
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2651
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2963
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3188

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2396

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1182471, upper bound: 0.1190137
time: 479.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1182577, upper bound: 0.1190012
time: 303.57 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 789.59 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 789.59
Output dim: 6, lower bound: -0.1182190, upper bound: 0.1190413
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 789.59
Output dim: 6, lower bound: -0.1182295, upper bound: 0.1190309
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 789.59
Output dim: 6, lower bound: -0.1182471, upper bound: 0.1190137
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 789.59
Output dim: 6, lower bound: -0.1182577, upper bound: 0.1190012
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 789.59
Output dim: 6, lower bound: -0.1190430, upper bound: 0.1182950
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 789.59
Output dim: 6, lower bound: -0.1190724, upper bound: 0.1182666

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 99.71 + 2164.11 = 2263.82 seconds
