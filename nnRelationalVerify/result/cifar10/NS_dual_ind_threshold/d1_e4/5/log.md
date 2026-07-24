## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 5)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.0343411245


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-4.1312189, -2.7349956, -4.1312189, -2.7349956, -0.5516378, 0.5516378)
1: (-3.5180688, -2.0475764, -3.5180688, -2.0475764, -0.6390442, 0.6390442)
2: (-1.6469839, -1.0061402, -1.6469839, -1.0061402, -0.4047842, 0.4047842)
3: (-0.3968609, -0.0245640, -0.3968609, -0.0245640, -0.2118566, 0.2118567)
4: (-1.9769460, -1.2035433, -1.9769460, -1.2035433, -0.2178566, 0.2178566)
5: (-0.9580898, -0.4939635, -0.9580898, -0.4939635, -0.2381067, 0.2381067)
6: (0.3418840, 0.9144647, 0.3418840, 0.9144647, -0.4361582, 0.4361582)
7: (-2.8870578, -2.1548257, -2.8870578, -2.1548257, -0.1774912, 0.1774912)
8: (-4.5607080, -3.0006542, -4.5607080, -3.0006542, -0.7997202, 0.7997200)
9: (-4.5105224, -3.1803586, -4.5105224, -3.1803586, -0.5872583, 0.5872583)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 8.25 + 267.62 = 275.87 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0343734, upper bound: 0.0343777

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 333
type: A, layer: 1, pos: 2123
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 2139
type: A, layer: 1, pos: 2544
type: A, layer: 1, pos: 2559
type: A, layer: 1, pos: 340
type: A, layer: 1, pos: 2109
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 2124
type: A, layer: 1, pos: 2106
type: A, layer: 1, pos: 2110
type: A, layer: 1, pos: 3482
type: A, layer: 1, pos: 381
type: A, layer: 1, pos: 2570
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 2392
type: A, layer: 1, pos: 384
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 2554
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 2402
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 3043
type: A, layer: 1, pos: 2357
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 3057
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 3131
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 2555
type: A, layer: 1, pos: 2548
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2627
type: A, layer: 1, pos: 2456
type: A, layer: 1, pos: 383
type: A, layer: 1, pos: 3117
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2517
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 2070
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 2398
type: A, layer: 1, pos: 506
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 2982
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 3002
type: A, layer: 1, pos: 3532
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 3097
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 3100
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 2408
type: A, layer: 1, pos: 2487
type: A, layer: 1, pos: 2403
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 3206
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 192
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 2537
type: A, layer: 1, pos: 3265
type: A, layer: 1, pos: 2160
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 2407
type: A, layer: 1, pos: 2307
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 2086
type: A, layer: 1, pos: 3249
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 3246
type: A, layer: 1, pos: 3084
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 3546
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 2055
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 3322
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 2085
type: A, layer: 1, pos: 2435
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 3103
type: A, layer: 1, pos: 2325
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 2413
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 3306
type: A, layer: 1, pos: 3250
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 3085
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 3003
type: A, layer: 1, pos: 2108
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 3335
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 2057
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 2223
type: A, layer: 1, pos: 3083
type: A, layer: 1, pos: 2672
type: A, layer: 1, pos: 3088
type: A, layer: 1, pos: 2206
type: A, layer: 1, pos: 2657
type: A, layer: 1, pos: 3320
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 2901
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 2433
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2046
type: A, layer: 1, pos: 2208
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 922
type: A, layer: 1, pos: 905
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 1036
type: A, layer: 1, pos: 1065
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1050
type: A, layer: 1, pos: 1108
type: A, layer: 1, pos: 1093
type: A, layer: 1, pos: 1051
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 149
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1064
type: A, layer: 1, pos: 1079
type: A, layer: 1, pos: 1094
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2369
type: A, layer: 1, pos: 2384
type: A, layer: 1, pos: 2399
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3044
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3142
type: A, layer: 1, pos: 3145
type: A, layer: 1, pos: 3146

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 333

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0341078, upper bound: 0.0343711
time: 325.35 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0343700, upper bound: 0.0343704
time: 170.29 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 495.71 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 495.71
Output dim: 6, lower bound: -0.0341078, upper bound: 0.0343711
NS_A2, status: Status.UNKNOWN, split count: 1, time: 495.71
Output dim: 6, lower bound: -0.0343700, upper bound: 0.0343704

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -4.1305494, -2.7337580, -4.1308751, -2.7349997, -0.5496554, 0.5510980
1: -3.5136518, -2.0525527, -3.5180681, -2.0520561, -0.6295495, 0.6333807
2: -1.6419739, -1.0071952, -1.6432310, -1.0061402, -0.4033579, 0.4001433
3: -0.3946956, -0.0266833, -0.3948276, -0.0245641, -0.2093897, 0.2077324
4: -1.9749694, -1.2052171, -1.9752036, -1.2035445, -0.2157569, 0.2144407
5: -0.9571939, -0.4947128, -0.9572254, -0.4939662, -0.2369535, 0.2360189
6: 0.3476975, 0.9094825, 0.3468895, 0.9144530, -0.4296286, 0.4255169
7: -2.8859115, -2.1555007, -2.8860762, -2.1548319, -0.1758209, 0.1749745
8: -4.5596261, -3.0022931, -4.5607047, -3.0021024, -0.7967960, 0.7981970
9: -4.5092592, -3.1821296, -4.5105228, -3.1818695, -0.5840252, 0.5850151

Time for backsubstitution: 6.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2123
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 2139
type: B, layer: 1, pos: 2544
type: B, layer: 1, pos: 2559
type: B, layer: 1, pos: 340
type: B, layer: 1, pos: 2109
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 2124
type: B, layer: 1, pos: 2106
type: B, layer: 1, pos: 2110
type: B, layer: 1, pos: 3482
type: B, layer: 1, pos: 381
type: B, layer: 1, pos: 2570
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 384
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2554
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 2402
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 2355
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 333
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3043
type: B, layer: 1, pos: 2357
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 3057
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3131
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 2555
type: B, layer: 1, pos: 2548
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2627
type: B, layer: 1, pos: 2456
type: B, layer: 1, pos: 383
type: B, layer: 1, pos: 3117
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2517
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 2070
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 506
type: B, layer: 1, pos: 2398
type: B, layer: 1, pos: 2982
type: B, layer: 1, pos: 2087
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 3002
type: B, layer: 1, pos: 3532
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 2036
type: B, layer: 1, pos: 3097
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 3100
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 2408
type: B, layer: 1, pos: 2487
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 3206
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 192
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 2537
type: B, layer: 1, pos: 3265
type: B, layer: 1, pos: 2160
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2407
type: B, layer: 1, pos: 2307
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 2086
type: B, layer: 1, pos: 3249
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 3246
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 3546
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 2055
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 3322
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 3082
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 2085
type: B, layer: 1, pos: 2435
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 2353
type: B, layer: 1, pos: 3103
type: B, layer: 1, pos: 2325
type: B, layer: 1, pos: 3081
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 3306
type: B, layer: 1, pos: 3250
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 3085
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 3003
type: B, layer: 1, pos: 2108
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 3335
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 2057
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 2223
type: B, layer: 1, pos: 3083
type: B, layer: 1, pos: 2672
type: B, layer: 1, pos: 3088
type: B, layer: 1, pos: 2206
type: B, layer: 1, pos: 2657
type: B, layer: 1, pos: 3320
type: B, layer: 1, pos: 2205
type: B, layer: 1, pos: 2901
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 2433
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2221
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2046
type: B, layer: 1, pos: 2208
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 922
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1036
type: B, layer: 1, pos: 1065
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1050
type: B, layer: 1, pos: 1108
type: B, layer: 1, pos: 1093
type: B, layer: 1, pos: 1051
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 149
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1064
type: B, layer: 1, pos: 1079
type: B, layer: 1, pos: 1094
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2369
type: B, layer: 1, pos: 2384
type: B, layer: 1, pos: 2399
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3044
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3142
type: B, layer: 1, pos: 3145
type: B, layer: 1, pos: 3146

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2123

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0340985, upper bound: 0.0340322
time: 246.54 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0340984, upper bound: 0.0343629
time: 124.22 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -4.1301622, -2.7349970, -4.1302633, -2.7349968, -0.5499673, 0.5497222
1: -3.5180686, -2.0475974, -3.5180686, -2.0475960, -0.6389703, 0.6328254
2: -1.6469204, -1.0061405, -1.6469262, -1.0061405, -0.4022012, 0.4035419
3: -0.3968395, -0.0245642, -0.3968416, -0.0245642, -0.2095261, 0.2118339
4: -1.9769404, -1.2035445, -1.9769411, -1.2035445, -0.2149390, 0.2178545
5: -0.9579676, -0.4939638, -0.9579793, -0.4939637, -0.2370279, 0.2380918
6: 0.3418952, 0.9144650, 0.3418941, 0.9144650, -0.4358658, 0.4364800
7: -2.8868029, -2.1548264, -2.8868275, -2.1548259, -0.1766047, 0.1774788
8: -4.5607080, -3.0008626, -4.5607080, -3.0008421, -0.7996885, 0.7973750
9: -4.5105228, -3.1805677, -4.5105228, -3.1805468, -0.5872119, 0.5868409

Time for backsubstitution: 6.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2123
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 2139
type: B, layer: 1, pos: 2544
type: B, layer: 1, pos: 2559
type: B, layer: 1, pos: 340
type: B, layer: 1, pos: 2109
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 2124
type: B, layer: 1, pos: 2106
type: B, layer: 1, pos: 2110
type: B, layer: 1, pos: 3482
type: B, layer: 1, pos: 381
type: B, layer: 1, pos: 2570
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 384
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2554
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 2402
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 2355
type: B, layer: 1, pos: 333
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 3043
type: B, layer: 1, pos: 2357
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 3057
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3131
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 2555
type: B, layer: 1, pos: 2548
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2627
type: B, layer: 1, pos: 2456
type: B, layer: 1, pos: 383
type: B, layer: 1, pos: 3117
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2517
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 2070
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 2398
type: B, layer: 1, pos: 506
type: B, layer: 1, pos: 2087
type: B, layer: 1, pos: 2982
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 3002
type: B, layer: 1, pos: 3532
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 2036
type: B, layer: 1, pos: 3097
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 3100
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 2408
type: B, layer: 1, pos: 2487
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 3206
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 192
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 2537
type: B, layer: 1, pos: 3265
type: B, layer: 1, pos: 2160
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2407
type: B, layer: 1, pos: 2307
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 2086
type: B, layer: 1, pos: 3249
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 3246
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 3546
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 2055
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 3322
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 3082
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 2085
type: B, layer: 1, pos: 2435
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 2353
type: B, layer: 1, pos: 3103
type: B, layer: 1, pos: 2325
type: B, layer: 1, pos: 3081
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 3306
type: B, layer: 1, pos: 3250
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 3085
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 3003
type: B, layer: 1, pos: 2108
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 3335
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 2057
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 2223
type: B, layer: 1, pos: 3083
type: B, layer: 1, pos: 2672
type: B, layer: 1, pos: 3088
type: B, layer: 1, pos: 2206
type: B, layer: 1, pos: 2657
type: B, layer: 1, pos: 3320
type: B, layer: 1, pos: 2205
type: B, layer: 1, pos: 2901
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 2433
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2221
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2046
type: B, layer: 1, pos: 2208
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 922
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1036
type: B, layer: 1, pos: 1065
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1050
type: B, layer: 1, pos: 1108
type: B, layer: 1, pos: 1093
type: B, layer: 1, pos: 1051
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 149
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1064
type: B, layer: 1, pos: 1079
type: B, layer: 1, pos: 1094
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2369
type: B, layer: 1, pos: 2384
type: B, layer: 1, pos: 2399
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3044
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3142
type: B, layer: 1, pos: 3145
type: B, layer: 1, pos: 3146

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2123

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0343613, upper bound: 0.0340371
time: 527.37 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0343596, upper bound: 0.0343641
time: 183.14 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 716.65 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 716.65
Output dim: 6, lower bound: -0.0340985, upper bound: 0.0340322
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 716.65
Output dim: 6, lower bound: -0.0340984, upper bound: 0.0343629
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 716.65
Output dim: 6, lower bound: -0.0343613, upper bound: 0.0340371
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 716.65
Output dim: 6, lower bound: -0.0343596, upper bound: 0.0343641

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 275.87 + 1589.26 = 1865.13 seconds
