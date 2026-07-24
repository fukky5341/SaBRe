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
execution time: IAR + RelationalAnalysis = 7.76 + 259.36 = 267.12 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0343734, upper bound: 0.0343777

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 333
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 3482
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2901
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 381
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 384
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 3322
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 607
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 1051
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 340
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 919
type: DSZ, layer: 1, pos: 920
type: DSZ, layer: 1, pos: 921
type: DSZ, layer: 1, pos: 922
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 924
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 934
type: DSZ, layer: 1, pos: 935
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 1035
type: DSZ, layer: 1, pos: 1036
type: DSZ, layer: 1, pos: 1064
type: DSZ, layer: 1, pos: 1079
type: DSZ, layer: 1, pos: 1093
type: DSZ, layer: 1, pos: 1094
type: DSZ, layer: 1, pos: 1108
type: DSZ, layer: 1, pos: 1109
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 1124
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2109
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2548
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3145
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 3265

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 333

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0341076, upper bound: 0.0343757
time: 20.60 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0343704, upper bound: 0.0341093
time: 38.47 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 59.14 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 59.14
Output dim: 6, lower bound: -0.0341076, upper bound: 0.0343757
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 59.14
Output dim: 6, lower bound: -0.0343704, upper bound: 0.0341093

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -4.1312189, -2.7349956, -4.1312189, -2.7349956, -0.5509128, 0.5508807
1: -3.5180688, -2.0475764, -3.5180688, -2.0475764, -0.6328542, 0.6327219
2: -1.6469839, -1.0061402, -1.6469839, -1.0061402, -0.4021007, 0.4022380
3: -0.3968609, -0.0245640, -0.3968609, -0.0245640, -0.2094776, 0.2095377
4: -1.9769460, -1.2035433, -1.9769460, -1.2035433, -0.2148665, 0.2149408
5: -0.9580898, -0.4939635, -0.9580898, -0.4939635, -0.2370215, 0.2370364
6: 0.3418840, 0.9144647, 0.3418840, 0.9144647, -0.4364932, 0.4364965
7: -2.8870578, -2.1548257, -2.8870578, -2.1548257, -0.1766032, 0.1766135
8: -4.5607080, -3.0006542, -4.5607080, -3.0006542, -0.7973881, 0.7973449
9: -4.5105224, -3.1803586, -4.5105224, -3.1803586, -0.5868862, 0.5868812

Time for backsubstitution: 6.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 3482
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2901
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 381
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 384
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 3322
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 607
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 1051
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 1035
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 340
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 919
type: DSZ, layer: 1, pos: 920
type: DSZ, layer: 1, pos: 921
type: DSZ, layer: 1, pos: 922
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 924
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 934
type: DSZ, layer: 1, pos: 935
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 1036
type: DSZ, layer: 1, pos: 1064
type: DSZ, layer: 1, pos: 1079
type: DSZ, layer: 1, pos: 1093
type: DSZ, layer: 1, pos: 1094
type: DSZ, layer: 1, pos: 1108
type: DSZ, layer: 1, pos: 1109
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 1124
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2109
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2548
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3145
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 3265

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2433

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0341054, upper bound: 0.0343746
time: 255.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0341058, upper bound: 0.0343733
time: 12.98 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -4.1312189, -2.7349956, -4.1312189, -2.7349956, -0.5508807, 0.5509127
1: -3.5180688, -2.0475764, -3.5180688, -2.0475764, -0.6327218, 0.6328542
2: -1.6469839, -1.0061402, -1.6469839, -1.0061402, -0.4022380, 0.4021008
3: -0.3968609, -0.0245640, -0.3968609, -0.0245640, -0.2095377, 0.2094776
4: -1.9769460, -1.2035433, -1.9769460, -1.2035433, -0.2149408, 0.2148665
5: -0.9580898, -0.4939635, -0.9580898, -0.4939635, -0.2370364, 0.2370215
6: 0.3418840, 0.9144647, 0.3418840, 0.9144647, -0.4364964, 0.4364932
7: -2.8870578, -2.1548257, -2.8870578, -2.1548257, -0.1766135, 0.1766032
8: -4.5607080, -3.0006542, -4.5607080, -3.0006542, -0.7973448, 0.7973881
9: -4.5105224, -3.1803586, -4.5105224, -3.1803586, -0.5868812, 0.5868862

Time for backsubstitution: 6.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 3482
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2901
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 381
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 384
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 3322
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 607
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 1051
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 1035
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 340
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 919
type: DSZ, layer: 1, pos: 920
type: DSZ, layer: 1, pos: 921
type: DSZ, layer: 1, pos: 922
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 924
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 934
type: DSZ, layer: 1, pos: 935
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 1036
type: DSZ, layer: 1, pos: 1064
type: DSZ, layer: 1, pos: 1079
type: DSZ, layer: 1, pos: 1093
type: DSZ, layer: 1, pos: 1094
type: DSZ, layer: 1, pos: 1108
type: DSZ, layer: 1, pos: 1109
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 1124
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2109
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2548
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3145
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 3265

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2433

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0343663, upper bound: 0.0341123
time: 17.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0343680, upper bound: 0.0341056
time: 81.56 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 104.73 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 104.73
Output dim: 6, lower bound: -0.0341054, upper bound: 0.0343746
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 104.73
Output dim: 6, lower bound: -0.0341058, upper bound: 0.0343733
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 104.73
Output dim: 6, lower bound: -0.0343663, upper bound: 0.0341123
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 104.73
Output dim: 6, lower bound: -0.0343680, upper bound: 0.0341056

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.1312189, -2.7349956, -4.1312189, -2.7349956, -0.5509101, 0.5508780
1: -3.5180688, -2.0475764, -3.5180688, -2.0475764, -0.6328514, 0.6327188
2: -1.6469839, -1.0061402, -1.6469839, -1.0061402, -0.4021004, 0.4022377
3: -0.3968609, -0.0245640, -0.3968609, -0.0245640, -0.2094776, 0.2095377
4: -1.9769460, -1.2035433, -1.9769460, -1.2035433, -0.2148640, 0.2149383
5: -0.9580898, -0.4939635, -0.9580898, -0.4939635, -0.2370213, 0.2370362
6: 0.3418840, 0.9144647, 0.3418840, 0.9144647, -0.4364930, 0.4364963
7: -2.8870578, -2.1548257, -2.8870578, -2.1548257, -0.1765985, 0.1766089
8: -4.5607080, -3.0006542, -4.5607080, -3.0006542, -0.7973853, 0.7973418
9: -4.5105224, -3.1803586, -4.5105224, -3.1803586, -0.5868835, 0.5868783

Time for backsubstitution: 6.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 3482
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2901
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 381
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 384
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 3322
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 607
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 1051
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 1035
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 340
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 919
type: DSZ, layer: 1, pos: 920
type: DSZ, layer: 1, pos: 921
type: DSZ, layer: 1, pos: 922
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 924
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 934
type: DSZ, layer: 1, pos: 935
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 1036
type: DSZ, layer: 1, pos: 1064
type: DSZ, layer: 1, pos: 1079
type: DSZ, layer: 1, pos: 1093
type: DSZ, layer: 1, pos: 1094
type: DSZ, layer: 1, pos: 1108
type: DSZ, layer: 1, pos: 1109
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 1124
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2109
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2548
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3145
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 3265

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2570

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0340824, upper bound: 0.0343292
time: 173.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0340631, upper bound: 0.0343498
time: 70.12 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.1312189, -2.7349956, -4.1312189, -2.7349956, -0.5509101, 0.5508780
1: -3.5180688, -2.0475764, -3.5180688, -2.0475764, -0.6328512, 0.6327190
2: -1.6469839, -1.0061402, -1.6469839, -1.0061402, -0.4021005, 0.4022377
3: -0.3968609, -0.0245640, -0.3968609, -0.0245640, -0.2094776, 0.2095377
4: -1.9769460, -1.2035433, -1.9769460, -1.2035433, -0.2148640, 0.2149383
5: -0.9580898, -0.4939635, -0.9580898, -0.4939635, -0.2370213, 0.2370362
6: 0.3418840, 0.9144647, 0.3418840, 0.9144647, -0.4364930, 0.4364963
7: -2.8870578, -2.1548257, -2.8870578, -2.1548257, -0.1765986, 0.1766088
8: -4.5607080, -3.0006542, -4.5607080, -3.0006542, -0.7973852, 0.7973421
9: -4.5105224, -3.1803586, -4.5105224, -3.1803586, -0.5868834, 0.5868785

Time for backsubstitution: 5.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 3482
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2901
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 381
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 384
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 3322
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 607
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 1051
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 1035
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 340
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 919
type: DSZ, layer: 1, pos: 920
type: DSZ, layer: 1, pos: 921
type: DSZ, layer: 1, pos: 922
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 924
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 934
type: DSZ, layer: 1, pos: 935
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 1036
type: DSZ, layer: 1, pos: 1064
type: DSZ, layer: 1, pos: 1079
type: DSZ, layer: 1, pos: 1093
type: DSZ, layer: 1, pos: 1094
type: DSZ, layer: 1, pos: 1108
type: DSZ, layer: 1, pos: 1109
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 1124
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2109
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2548
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3145
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 3265

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2570

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0340854, upper bound: 0.0343332
time: 6.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0340670, upper bound: 0.0343480
time: 21.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.1312189, -2.7349956, -4.1312189, -2.7349956, -0.5508780, 0.5509101
1: -3.5180688, -2.0475764, -3.5180688, -2.0475764, -0.6327190, 0.6328511
2: -1.6469839, -1.0061402, -1.6469839, -1.0061402, -0.4022377, 0.4021004
3: -0.3968609, -0.0245640, -0.3968609, -0.0245640, -0.2095377, 0.2094776
4: -1.9769460, -1.2035433, -1.9769460, -1.2035433, -0.2149383, 0.2148640
5: -0.9580898, -0.4939635, -0.9580898, -0.4939635, -0.2370362, 0.2370213
6: 0.3418840, 0.9144647, 0.3418840, 0.9144647, -0.4364963, 0.4364930
7: -2.8870578, -2.1548257, -2.8870578, -2.1548257, -0.1766088, 0.1765986
8: -4.5607080, -3.0006542, -4.5607080, -3.0006542, -0.7973421, 0.7973852
9: -4.5105224, -3.1803586, -4.5105224, -3.1803586, -0.5868785, 0.5868833

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 3482
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2901
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 381
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 384
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 3322
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 607
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 1051
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 1035
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 340
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 919
type: DSZ, layer: 1, pos: 920
type: DSZ, layer: 1, pos: 921
type: DSZ, layer: 1, pos: 922
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 924
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 934
type: DSZ, layer: 1, pos: 935
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 1036
type: DSZ, layer: 1, pos: 1064
type: DSZ, layer: 1, pos: 1079
type: DSZ, layer: 1, pos: 1093
type: DSZ, layer: 1, pos: 1094
type: DSZ, layer: 1, pos: 1108
type: DSZ, layer: 1, pos: 1109
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 1124
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2109
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2548
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3145
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 3265

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2570

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0343444, upper bound: 0.0340676
time: 143.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0343264, upper bound: 0.0340924
time: 9.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.1312189, -2.7349956, -4.1312189, -2.7349956, -0.5508780, 0.5509101
1: -3.5180688, -2.0475764, -3.5180688, -2.0475764, -0.6327189, 0.6328514
2: -1.6469839, -1.0061402, -1.6469839, -1.0061402, -0.4022377, 0.4021004
3: -0.3968609, -0.0245640, -0.3968609, -0.0245640, -0.2095377, 0.2094776
4: -1.9769460, -1.2035433, -1.9769460, -1.2035433, -0.2149384, 0.2148640
5: -0.9580898, -0.4939635, -0.9580898, -0.4939635, -0.2370362, 0.2370213
6: 0.3418840, 0.9144647, 0.3418840, 0.9144647, -0.4364963, 0.4364930
7: -2.8870578, -2.1548257, -2.8870578, -2.1548257, -0.1766089, 0.1765985
8: -4.5607080, -3.0006542, -4.5607080, -3.0006542, -0.7973419, 0.7973852
9: -4.5105224, -3.1803586, -4.5105224, -3.1803586, -0.5868782, 0.5868835

Time for backsubstitution: 6.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 3482
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2901
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 381
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 384
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 3322
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 607
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 1051
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 1035
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 340
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 919
type: DSZ, layer: 1, pos: 920
type: DSZ, layer: 1, pos: 921
type: DSZ, layer: 1, pos: 922
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 924
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 934
type: DSZ, layer: 1, pos: 935
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 1036
type: DSZ, layer: 1, pos: 1064
type: DSZ, layer: 1, pos: 1079
type: DSZ, layer: 1, pos: 1093
type: DSZ, layer: 1, pos: 1094
type: DSZ, layer: 1, pos: 1108
type: DSZ, layer: 1, pos: 1109
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 1124
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2109
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2548
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3145
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 3265

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2570

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0343470, upper bound: 0.0340679
time: 101.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0343280, upper bound: 0.0340840
time: 202.99 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 310.95 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 310.95
Output dim: 6, lower bound: -0.0340824, upper bound: 0.0343292
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 310.95
Output dim: 6, lower bound: -0.0340631, upper bound: 0.0343498
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 310.95
Output dim: 6, lower bound: -0.0340854, upper bound: 0.0343332
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 310.95
Output dim: 6, lower bound: -0.0340670, upper bound: 0.0343480
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 310.95
Output dim: 6, lower bound: -0.0343444, upper bound: 0.0340676
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 310.95
Output dim: 6, lower bound: -0.0343264, upper bound: 0.0340924
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 310.95
Output dim: 6, lower bound: -0.0343470, upper bound: 0.0340679
DS_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 310.95
Output dim: 6, lower bound: -0.0343280, upper bound: 0.0340840

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.1312189, -2.7349956, -4.1312189, -2.7349956, -0.5411571, 0.5410546
1: -3.5180688, -2.0475764, -3.5180688, -2.0475764, -0.6231354, 0.6229149
2: -1.6469839, -1.0061402, -1.6469839, -1.0061402, -0.4020502, 0.4021851
3: -0.3968609, -0.0245640, -0.3968609, -0.0245640, -0.2088848, 0.2089466
4: -1.9769460, -1.2035433, -1.9769460, -1.2035433, -0.2145065, 0.2145693
5: -0.9580898, -0.4939635, -0.9580898, -0.4939635, -0.2364156, 0.2364364
6: 0.3418840, 0.9144647, 0.3418840, 0.9144647, -0.4362118, 0.4362167
7: -2.8870578, -2.1548257, -2.8870578, -2.1548257, -0.1764944, 0.1765010
8: -4.5607080, -3.0006542, -4.5607080, -3.0006542, -0.7923687, 0.7922546
9: -4.5105224, -3.1803586, -4.5105224, -3.1803586, -0.5805244, 0.5804529

Time for backsubstitution: 5.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 3482
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2901
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 381
type: DSZ, layer: 1, pos: 384
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 3322
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 607
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 1051
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 1035
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 340
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 919
type: DSZ, layer: 1, pos: 920
type: DSZ, layer: 1, pos: 921
type: DSZ, layer: 1, pos: 922
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 924
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 934
type: DSZ, layer: 1, pos: 935
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 1036
type: DSZ, layer: 1, pos: 1064
type: DSZ, layer: 1, pos: 1079
type: DSZ, layer: 1, pos: 1093
type: DSZ, layer: 1, pos: 1094
type: DSZ, layer: 1, pos: 1108
type: DSZ, layer: 1, pos: 1109
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 1124
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2109
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2548
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3145
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 3265

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3110

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0340609, upper bound: 0.0343503
time: 201.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0340612, upper bound: 0.0343484
time: 25.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.1312189, -2.7349956, -4.1312189, -2.7349956, -0.5411571, 0.5410548
1: -3.5180688, -2.0475764, -3.5180688, -2.0475764, -0.6231353, 0.6229150
2: -1.6469839, -1.0061402, -1.6469839, -1.0061402, -0.4020502, 0.4021851
3: -0.3968609, -0.0245640, -0.3968609, -0.0245640, -0.2088848, 0.2089466
4: -1.9769460, -1.2035433, -1.9769460, -1.2035433, -0.2145066, 0.2145693
5: -0.9580898, -0.4939635, -0.9580898, -0.4939635, -0.2364156, 0.2364364
6: 0.3418840, 0.9144647, 0.3418840, 0.9144647, -0.4362118, 0.4362167
7: -2.8870578, -2.1548257, -2.8870578, -2.1548257, -0.1764945, 0.1765009
8: -4.5607080, -3.0006542, -4.5607080, -3.0006542, -0.7923685, 0.7922546
9: -4.5105224, -3.1803586, -4.5105224, -3.1803586, -0.5805244, 0.5804530

Time for backsubstitution: 5.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 3482
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2901
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 381
type: DSZ, layer: 1, pos: 384
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 3322
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 607
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 1051
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 1035
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 340
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 919
type: DSZ, layer: 1, pos: 920
type: DSZ, layer: 1, pos: 921
type: DSZ, layer: 1, pos: 922
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 924
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 934
type: DSZ, layer: 1, pos: 935
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 1036
type: DSZ, layer: 1, pos: 1064
type: DSZ, layer: 1, pos: 1079
type: DSZ, layer: 1, pos: 1093
type: DSZ, layer: 1, pos: 1094
type: DSZ, layer: 1, pos: 1108
type: DSZ, layer: 1, pos: 1109
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 1124
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2109
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2548
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3145
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 3265

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3110

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0340619, upper bound: 0.0343483
time: 9.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0340635, upper bound: 0.0343485
time: 17.37 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.1312189, -2.7349956, -4.1312189, -2.7349956, -0.5410548, 0.5411571
1: -3.5180688, -2.0475764, -3.5180688, -2.0475764, -0.6229150, 0.6231353
2: -1.6469839, -1.0061402, -1.6469839, -1.0061402, -0.4021852, 0.4020502
3: -0.3968609, -0.0245640, -0.3968609, -0.0245640, -0.2089466, 0.2088848
4: -1.9769460, -1.2035433, -1.9769460, -1.2035433, -0.2145693, 0.2145066
5: -0.9580898, -0.4939635, -0.9580898, -0.4939635, -0.2364364, 0.2364156
6: 0.3418840, 0.9144647, 0.3418840, 0.9144647, -0.4362167, 0.4362119
7: -2.8870578, -2.1548257, -2.8870578, -2.1548257, -0.1765009, 0.1764945
8: -4.5607080, -3.0006542, -4.5607080, -3.0006542, -0.7922547, 0.7923685
9: -4.5105224, -3.1803586, -4.5105224, -3.1803586, -0.5804529, 0.5805244

Time for backsubstitution: 5.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 3482
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2901
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 381
type: DSZ, layer: 1, pos: 384
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 3322
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 607
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 1051
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 1035
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 340
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 919
type: DSZ, layer: 1, pos: 920
type: DSZ, layer: 1, pos: 921
type: DSZ, layer: 1, pos: 922
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 924
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 934
type: DSZ, layer: 1, pos: 935
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 1036
type: DSZ, layer: 1, pos: 1064
type: DSZ, layer: 1, pos: 1079
type: DSZ, layer: 1, pos: 1093
type: DSZ, layer: 1, pos: 1094
type: DSZ, layer: 1, pos: 1108
type: DSZ, layer: 1, pos: 1109
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 1124
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2109
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2548
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3145
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 3265

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3110

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0343429, upper bound: 0.0340662
time: 243.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0343435, upper bound: 0.0340647
time: 52.46 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 302.08 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 302.08
Output dim: 6, lower bound: -0.0340609, upper bound: 0.0343503
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 302.08
Output dim: 6, lower bound: -0.0340612, upper bound: 0.0343484
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 302.08
Output dim: 6, lower bound: -0.0340619, upper bound: 0.0343483
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 302.08
Output dim: 6, lower bound: -0.0340635, upper bound: 0.0343485
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 302.08
Output dim: 6, lower bound: -0.0343429, upper bound: 0.0340662
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 302.08
Output dim: 6, lower bound: -0.0343435, upper bound: 0.0340647
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 302.08
Output dim: 6, lower bound: -0.0343470, upper bound: 0.0340679

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 267.12 + 1759.82 = 2026.95 seconds
