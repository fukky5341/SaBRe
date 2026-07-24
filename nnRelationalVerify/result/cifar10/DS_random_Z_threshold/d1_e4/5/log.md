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
execution time: IAR + RelationalAnalysis = 8.26 + 262.90 = 271.16 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0343734, upper bound: 0.0343777

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 922
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 924
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 1035
type: DSZ, layer: 1, pos: 2901
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 1079
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 935
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 919
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 384
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2548
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 1094
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 381
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 920
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 1124
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 934
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 607
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 333
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 1109
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2109
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 3322
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 3482
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 1108
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 1093
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 1064
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 921
type: DSZ, layer: 1, pos: 3145
type: DSZ, layer: 1, pos: 1036
type: DSZ, layer: 1, pos: 1051
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 340
type: DSZ, layer: 1, pos: 2552

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3029

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0343759, upper bound: 0.0343772
time: 111.14 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0343759, upper bound: 0.0343790
time: 169.17 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 280.32 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 280.32
Output dim: 6, lower bound: -0.0343759, upper bound: 0.0343772
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 280.32
Output dim: 6, lower bound: -0.0343759, upper bound: 0.0343790

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -4.1312189, -2.7349956, -4.1312189, -2.7349956, -0.5516378, 0.5516378
1: -3.5180688, -2.0475764, -3.5180688, -2.0475764, -0.6390442, 0.6390442
2: -1.6469839, -1.0061402, -1.6469839, -1.0061402, -0.4047842, 0.4047842
3: -0.3968609, -0.0245640, -0.3968609, -0.0245640, -0.2118566, 0.2118567
4: -1.9769460, -1.2035433, -1.9769460, -1.2035433, -0.2178566, 0.2178566
5: -0.9580898, -0.4939635, -0.9580898, -0.4939635, -0.2381067, 0.2381067
6: 0.3418840, 0.9144647, 0.3418840, 0.9144647, -0.4361582, 0.4361582
7: -2.8870578, -2.1548257, -2.8870578, -2.1548257, -0.1774912, 0.1774912
8: -4.5607080, -3.0006542, -4.5607080, -3.0006542, -0.7997202, 0.7997200
9: -4.5105224, -3.1803586, -4.5105224, -3.1803586, -0.5872583, 0.5872583

Time for backsubstitution: 6.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 921
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 1094
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 2901
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 1079
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 1064
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 920
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 1036
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 922
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 1093
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 1124
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 381
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 924
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 3322
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 333
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 607
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 935
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 3482
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 1035
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 384
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 1108
type: DSZ, layer: 1, pos: 3145
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 919
type: DSZ, layer: 1, pos: 1051
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 340
type: DSZ, layer: 1, pos: 934
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2109
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 1109
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2548
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 3002

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2485

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0343704, upper bound: 0.0343743
time: 120.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0343698, upper bound: 0.0343751
time: 11.98 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -4.1312189, -2.7349956, -4.1312189, -2.7349956, -0.5516378, 0.5516378
1: -3.5180688, -2.0475764, -3.5180688, -2.0475764, -0.6390442, 0.6390442
2: -1.6469839, -1.0061402, -1.6469839, -1.0061402, -0.4047842, 0.4047842
3: -0.3968609, -0.0245640, -0.3968609, -0.0245640, -0.2118566, 0.2118567
4: -1.9769460, -1.2035433, -1.9769460, -1.2035433, -0.2178566, 0.2178566
5: -0.9580898, -0.4939635, -0.9580898, -0.4939635, -0.2381067, 0.2381067
6: 0.3418840, 0.9144647, 0.3418840, 0.9144647, -0.4361582, 0.4361582
7: -2.8870578, -2.1548257, -2.8870578, -2.1548257, -0.1774912, 0.1774912
8: -4.5607080, -3.0006542, -4.5607080, -3.0006542, -0.7997202, 0.7997200
9: -4.5105224, -3.1803586, -4.5105224, -3.1803586, -0.5872583, 0.5872583

Time for backsubstitution: 6.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 2548
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 1109
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 921
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 340
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 1079
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 381
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 1093
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 1124
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 1035
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 1108
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3322
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 3145
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 1064
type: DSZ, layer: 1, pos: 607
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 1051
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 1036
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 935
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 919
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2901
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2109
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3482
type: DSZ, layer: 1, pos: 333
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 922
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 384
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 934
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 924
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 920
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 1094
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 1123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1122

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0343738, upper bound: 0.0343773
time: 141.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0343738, upper bound: 0.0343790
time: 120.06 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 267.84 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 267.84
Output dim: 6, lower bound: -0.0343704, upper bound: 0.0343743
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 267.84
Output dim: 6, lower bound: -0.0343698, upper bound: 0.0343751
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 267.84
Output dim: 6, lower bound: -0.0343738, upper bound: 0.0343773
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 267.84
Output dim: 6, lower bound: -0.0343738, upper bound: 0.0343790

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.1312189, -2.7349956, -4.1312189, -2.7349956, -0.5516269, 0.5516270
1: -3.5180688, -2.0475764, -3.5180688, -2.0475764, -0.6389892, 0.6389902
2: -1.6469839, -1.0061402, -1.6469839, -1.0061402, -0.4047729, 0.4047730
3: -0.3968609, -0.0245640, -0.3968609, -0.0245640, -0.2118530, 0.2118531
4: -1.9769460, -1.2035433, -1.9769460, -1.2035433, -0.2178266, 0.2178265
5: -0.9580898, -0.4939635, -0.9580898, -0.4939635, -0.2380908, 0.2380911
6: 0.3418840, 0.9144647, 0.3418840, 0.9144647, -0.4361036, 0.4361023
7: -2.8870578, -2.1548257, -2.8870578, -2.1548257, -0.1773913, 0.1773951
8: -4.5607080, -3.0006542, -4.5607080, -3.0006542, -0.7996505, 0.7996538
9: -4.5105224, -3.1803586, -4.5105224, -3.1803586, -0.5872254, 0.5872253

Time for backsubstitution: 6.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 384
type: DSZ, layer: 1, pos: 920
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 381
type: DSZ, layer: 1, pos: 921
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 333
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 935
type: DSZ, layer: 1, pos: 2548
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 607
type: DSZ, layer: 1, pos: 340
type: DSZ, layer: 1, pos: 1093
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 934
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 3482
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 1094
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 1124
type: DSZ, layer: 1, pos: 2109
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 1035
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 1036
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 1064
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 1079
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 1051
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2901
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 919
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 1108
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 3145
type: DSZ, layer: 1, pos: 922
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 3322
type: DSZ, layer: 1, pos: 924
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 1109
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2552

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2235

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0343710, upper bound: 0.0343730
time: 173.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0343710, upper bound: 0.0343744
time: 15.95 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.1312189, -2.7349956, -4.1312189, -2.7349956, -0.5516271, 0.5516269
1: -3.5180688, -2.0475764, -3.5180688, -2.0475764, -0.6389902, 0.6389892
2: -1.6469839, -1.0061402, -1.6469839, -1.0061402, -0.4047730, 0.4047730
3: -0.3968609, -0.0245640, -0.3968609, -0.0245640, -0.2118531, 0.2118531
4: -1.9769460, -1.2035433, -1.9769460, -1.2035433, -0.2178264, 0.2178267
5: -0.9580898, -0.4939635, -0.9580898, -0.4939635, -0.2380911, 0.2380908
6: 0.3418840, 0.9144647, 0.3418840, 0.9144647, -0.4361023, 0.4361036
7: -2.8870578, -2.1548257, -2.8870578, -2.1548257, -0.1773951, 0.1773913
8: -4.5607080, -3.0006542, -4.5607080, -3.0006542, -0.7996538, 0.7996507
9: -4.5105224, -3.1803586, -4.5105224, -3.1803586, -0.5872254, 0.5872253

Time for backsubstitution: 6.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 1094
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 3482
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 2109
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 922
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 1093
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 919
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 1079
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 1108
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 1064
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 3145
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 921
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 384
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 920
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 1051
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2548
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 1109
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 1124
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 381
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 935
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2901
type: DSZ, layer: 1, pos: 1035
type: DSZ, layer: 1, pos: 607
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 924
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3322
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 934
type: DSZ, layer: 1, pos: 340
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 1036
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 333
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 918

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 945

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0343684, upper bound: 0.0343764
time: 17.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0343682, upper bound: 0.0343678
time: 399.96 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.1312189, -2.7349956, -4.1312189, -2.7349956, -0.5516378, 0.5516378
1: -3.5180688, -2.0475764, -3.5180688, -2.0475764, -0.6390442, 0.6390442
2: -1.6469839, -1.0061402, -1.6469839, -1.0061402, -0.4047842, 0.4047842
3: -0.3968609, -0.0245640, -0.3968609, -0.0245640, -0.2118566, 0.2118567
4: -1.9769460, -1.2035433, -1.9769460, -1.2035433, -0.2178566, 0.2178566
5: -0.9580898, -0.4939635, -0.9580898, -0.4939635, -0.2381067, 0.2381067
6: 0.3418840, 0.9144647, 0.3418840, 0.9144647, -0.4361582, 0.4361582
7: -2.8870578, -2.1548257, -2.8870578, -2.1548257, -0.1774912, 0.1774912
8: -4.5607080, -3.0006542, -4.5607080, -3.0006542, -0.7997202, 0.7997200
9: -4.5105224, -3.1803586, -4.5105224, -3.1803586, -0.5872583, 0.5872583

Time for backsubstitution: 6.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 384
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 1093
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 381
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 924
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 920
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 333
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 3482
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2109
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 1035
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 340
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 1036
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 1108
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 934
type: DSZ, layer: 1, pos: 922
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 1094
type: DSZ, layer: 1, pos: 1109
type: DSZ, layer: 1, pos: 607
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 1051
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 1124
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2548
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 1079
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 921
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 3322
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 3145
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 919
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 1064
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 935
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 2901
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 737

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 886

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0343742, upper bound: 0.0343748
time: 36.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0343742, upper bound: 0.0343747
time: 33.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.1312189, -2.7349956, -4.1312189, -2.7349956, -0.5516378, 0.5516378
1: -3.5180688, -2.0475764, -3.5180688, -2.0475764, -0.6390442, 0.6390442
2: -1.6469839, -1.0061402, -1.6469839, -1.0061402, -0.4047842, 0.4047842
3: -0.3968609, -0.0245640, -0.3968609, -0.0245640, -0.2118566, 0.2118567
4: -1.9769460, -1.2035433, -1.9769460, -1.2035433, -0.2178566, 0.2178566
5: -0.9580898, -0.4939635, -0.9580898, -0.4939635, -0.2381067, 0.2381067
6: 0.3418840, 0.9144647, 0.3418840, 0.9144647, -0.4361582, 0.4361582
7: -2.8870578, -2.1548257, -2.8870578, -2.1548257, -0.1774912, 0.1774912
8: -4.5607080, -3.0006542, -4.5607080, -3.0006542, -0.7997202, 0.7997200
9: -4.5105224, -3.1803586, -4.5105224, -3.1803586, -0.5872583, 0.5872583

Time for backsubstitution: 6.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 1109
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 1123
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 1064
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 1094
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 3043
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 3131
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 333
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 935
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 606
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 945
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2109
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 921
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 3482
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 1035
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 920
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 922
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 1050
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 381
type: DSZ, layer: 1, pos: 1108
type: DSZ, layer: 1, pos: 1124
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 3532
type: DSZ, layer: 1, pos: 1093
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 3322
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 919
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 340
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 3146
type: DSZ, layer: 1, pos: 384
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 1036
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 924
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 607
type: DSZ, layer: 1, pos: 1079
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 1051
type: DSZ, layer: 1, pos: 2901
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2548
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 934
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 3546
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 3145
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 3265
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2485

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 685

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0343722, upper bound: 0.0343777
time: 210.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0343737, upper bound: 0.0343743
time: 91.04 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 308.12 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 308.12
Output dim: 6, lower bound: -0.0343710, upper bound: 0.0343730
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 308.12
Output dim: 6, lower bound: -0.0343710, upper bound: 0.0343744
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 308.12
Output dim: 6, lower bound: -0.0343684, upper bound: 0.0343764
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 308.12
Output dim: 6, lower bound: -0.0343682, upper bound: 0.0343678
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 308.12
Output dim: 6, lower bound: -0.0343742, upper bound: 0.0343748
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 308.12
Output dim: 6, lower bound: -0.0343742, upper bound: 0.0343747
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 308.12
Output dim: 6, lower bound: -0.0343722, upper bound: 0.0343777
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 308.12
Output dim: 6, lower bound: -0.0343737, upper bound: 0.0343743

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 271.16 + 1689.17 = 1960.33 seconds
