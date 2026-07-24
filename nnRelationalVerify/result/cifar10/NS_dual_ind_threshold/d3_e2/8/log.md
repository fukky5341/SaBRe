## Execution arguments:
Dataset: Dataset.CIFAR10
Network: onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 8)
Time budget: 7200 seconds
Split limit: 100
Threshold: 0.2537563896


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-4.3332100, -1.7664787, -4.3332100, -1.7664787, -0.9647807, 0.9647807)
1: (-3.5020158, -0.5655890, -3.5020158, -0.5655890, -1.2024612, 1.2024610)
2: (-1.8742059, -1.2430845, -1.8742059, -1.2430845, -0.2156219, 0.2156219)
3: (-0.3931611, 0.0556504, -0.3931611, 0.0556504, -0.3053983, 0.3053983)
4: (-0.9882242, 0.1546044, -0.9882242, 0.1546044, -0.9380934, 0.9380935)
5: (-0.6528150, -0.1970142, -0.6528150, -0.1970142, -0.2471910, 0.2471910)
6: (-2.7923806, -1.1410340, -2.7923806, -1.1410340, -0.5087215, 0.5087214)
7: (-0.7873561, 0.3647786, -0.7873561, 0.3647786, -1.0119059, 1.0119059)
8: (-4.1237440, -0.9955738, -4.1237440, -0.9955738, -1.2521666, 1.2521667)
9: (-4.9525437, -2.2417867, -4.9525437, -2.2417867, -1.0728309, 1.0728309)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 5.04 + 337.71 = 342.75 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.2540104, upper bound: 0.2540117

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 276
type: A, layer: 1, pos: 396
type: A, layer: 1, pos: 3097
type: A, layer: 1, pos: 2452
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 334
type: A, layer: 1, pos: 261
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 2407
type: A, layer: 1, pos: 337
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 3550
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 364
type: A, layer: 1, pos: 3425
type: A, layer: 1, pos: 350
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 2651
type: A, layer: 1, pos: 2677
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 385
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 333
type: A, layer: 1, pos: 379
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 332
type: A, layer: 1, pos: 3549
type: A, layer: 1, pos: 275
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 2584
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 192
type: A, layer: 1, pos: 2108
type: A, layer: 1, pos: 3503
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 3484
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 2109
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 2209
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2970
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 3049
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 395
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3319
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3002
type: A, layer: 1, pos: 3481
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 2217
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 2338
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 2312
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 3279
type: A, layer: 1, pos: 3322
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 2113
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 2127
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 3186
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 2313
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 2106
type: A, layer: 1, pos: 2102
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 2626
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 2086
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 2258
type: A, layer: 1, pos: 2259
type: A, layer: 1, pos: 2446
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 2181
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 2976
type: A, layer: 1, pos: 2206
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 2251
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 2250
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2940
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2480
type: A, layer: 1, pos: 2092
type: A, layer: 1, pos: 2595
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 206
type: A, layer: 1, pos: 2068
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2100
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 2315
type: A, layer: 1, pos: 357
type: A, layer: 1, pos: 2507
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2254
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 356
type: A, layer: 1, pos: 3005
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3326
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2260
type: A, layer: 1, pos: 2484
type: A, layer: 1, pos: 2078
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 2118
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 2262
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 2110
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 2263
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 149
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 404
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 2084
type: A, layer: 1, pos: 2099
type: A, layer: 1, pos: 2144
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 2624
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3135
type: A, layer: 1, pos: 3136
type: A, layer: 1, pos: 3142
type: A, layer: 1, pos: 3143
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3149
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3591

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 276

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2533548, upper bound: 0.2539502
time: 197.19 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2540056, upper bound: 0.2540059
time: 301.36 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 498.67 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 498.67
Output dim: 4, lower bound: -0.2533548, upper bound: 0.2539502
IS_A2, status: Status.UNKNOWN, split count: 1, time: 498.67
Output dim: 4, lower bound: -0.2540056, upper bound: 0.2540059

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -4.3303614, -1.7664831, -4.3307285, -1.7664829, -0.9621660, 0.9624461
1: -3.5014367, -0.5655942, -3.5015831, -0.5655928, -1.2018054, 1.2019556
2: -1.8741231, -1.2480179, -1.8741317, -1.2471018, -0.2115309, 0.2106648
3: -0.3920907, 0.0555136, -0.3922355, 0.0555300, -0.3041323, 0.3042993
4: -0.9814273, 0.1546015, -0.9826717, 0.1546023, -0.9322866, 0.9332354
5: -0.6528058, -0.1997054, -0.6528075, -0.1992135, -0.2447381, 0.2443119
6: -2.7875242, -1.1410506, -2.7882867, -1.1410491, -0.5031639, 0.5040892
7: -0.7871017, 0.3601484, -0.7871305, 0.3609633, -1.0075052, 1.0067631
8: -4.1215191, -0.9956119, -4.1218429, -0.9956090, -1.2504171, 1.2505915
9: -4.9525385, -2.2423420, -4.9525404, -2.2422180, -1.0716538, 1.0714895

Time for backsubstitution: 4.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 396
type: B, layer: 1, pos: 3097
type: B, layer: 1, pos: 2452
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 334
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 2407
type: B, layer: 1, pos: 337
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 3550
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 261
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 364
type: B, layer: 1, pos: 3425
type: B, layer: 1, pos: 350
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 2651
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 2231
type: B, layer: 1, pos: 385
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 333
type: B, layer: 1, pos: 379
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 332
type: B, layer: 1, pos: 3549
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 275
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 2584
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 2600
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 276
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 192
type: B, layer: 1, pos: 2108
type: B, layer: 1, pos: 3503
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 3484
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 2109
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 2209
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2970
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 3049
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 395
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3319
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3002
type: B, layer: 1, pos: 3481
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 2217
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2338
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2312
type: B, layer: 1, pos: 145
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 3322
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2056
type: B, layer: 1, pos: 2113
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 2127
type: B, layer: 1, pos: 2087
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 2313
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2106
type: B, layer: 1, pos: 2102
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 2626
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 2086
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 3186
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 2258
type: B, layer: 1, pos: 2259
type: B, layer: 1, pos: 2446
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 2181
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 2976
type: B, layer: 1, pos: 2206
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 2251
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 2205
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 2250
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2940
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 2480
type: B, layer: 1, pos: 2092
type: B, layer: 1, pos: 2595
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 206
type: B, layer: 1, pos: 2068
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 2100
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 2315
type: B, layer: 1, pos: 357
type: B, layer: 1, pos: 2507
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2254
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 2052
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 356
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 3005
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3326
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2539
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2260
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 2078
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 2118
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 2051
type: B, layer: 1, pos: 2262
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2110
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 2263
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 149
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 404
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 2084
type: B, layer: 1, pos: 2099
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 2624
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3135
type: B, layer: 1, pos: 3136
type: B, layer: 1, pos: 3142
type: B, layer: 1, pos: 3143
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3149
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3591

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 396

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.2532689, upper bound: 0.2533356
time: 29.00 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2533377, upper bound: 0.2539428
time: 157.38 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -4.3331118, -1.7628967, -4.3328695, -1.7664787, -0.9644146, 0.9687043
1: -3.5015621, -0.5658026, -3.5007305, -0.5655890, -1.2023920, 1.2010088
2: -1.8814406, -1.2430372, -1.8741775, -1.2430882, -0.2228023, 0.2152374
3: -0.3932819, 0.0577744, -0.3931362, 0.0556358, -0.3054735, 0.3074746
4: -0.9896809, 0.1639230, -0.9881492, 0.1545995, -0.9407603, 0.9473589
5: -0.6566840, -0.1960614, -0.6528133, -0.1970266, -0.2513817, 0.2481777
6: -2.7929931, -1.1331664, -2.7922006, -1.1410412, -0.5091517, 0.5172141
7: -0.7942684, 0.3650010, -0.7873283, 0.3647631, -1.0179970, 1.0115770
8: -4.1251040, -0.9920235, -4.1235499, -0.9955873, -1.2550565, 1.2555789
9: -4.9532170, -2.2416368, -4.9525361, -2.2418048, -1.0744514, 1.0715992

Time for backsubstitution: 4.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 396
type: B, layer: 1, pos: 3097
type: B, layer: 1, pos: 2452
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 334
type: B, layer: 1, pos: 261
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 2407
type: B, layer: 1, pos: 337
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 3550
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 364
type: B, layer: 1, pos: 3425
type: B, layer: 1, pos: 350
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 2651
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 2231
type: B, layer: 1, pos: 385
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 276
type: B, layer: 1, pos: 333
type: B, layer: 1, pos: 379
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 332
type: B, layer: 1, pos: 3549
type: B, layer: 1, pos: 275
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 2584
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 2600
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 192
type: B, layer: 1, pos: 2108
type: B, layer: 1, pos: 3503
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 3484
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 2109
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 2209
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2970
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 3049
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 395
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3319
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3002
type: B, layer: 1, pos: 3481
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 2217
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2338
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2312
type: B, layer: 1, pos: 145
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 3322
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2056
type: B, layer: 1, pos: 2113
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 2127
type: B, layer: 1, pos: 2087
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 3186
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 2313
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2106
type: B, layer: 1, pos: 2102
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 2626
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 2086
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 2258
type: B, layer: 1, pos: 2259
type: B, layer: 1, pos: 2446
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 2181
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 2976
type: B, layer: 1, pos: 2206
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 2251
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 2205
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 2250
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2940
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 2480
type: B, layer: 1, pos: 2092
type: B, layer: 1, pos: 2595
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 206
type: B, layer: 1, pos: 2068
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 2100
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 2315
type: B, layer: 1, pos: 357
type: B, layer: 1, pos: 2507
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2254
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 2052
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 356
type: B, layer: 1, pos: 3005
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3326
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2539
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2260
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 2078
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 2118
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 2051
type: B, layer: 1, pos: 2262
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2110
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 2263
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 149
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 404
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 2084
type: B, layer: 1, pos: 2099
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 2624
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3135
type: B, layer: 1, pos: 3136
type: B, layer: 1, pos: 3142
type: B, layer: 1, pos: 3143
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3149
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3591

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 396

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2539386, upper bound: 0.2534005
time: 527.47 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2540011, upper bound: 0.2540020
time: 592.27 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 1124.25 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 1124.25
Output dim: 4, lower bound: -0.2532689, upper bound: 0.2533356
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 1124.25
Output dim: 4, lower bound: -0.2533377, upper bound: 0.2539428
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 1124.25
Output dim: 4, lower bound: -0.2539386, upper bound: 0.2534005
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 1124.25
Output dim: 4, lower bound: -0.2540011, upper bound: 0.2540020

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -4.3303423, -1.7664959, -4.3311958, -1.7664951, -0.9617560, 0.9621332
1: -3.4983726, -0.5655937, -3.4980602, -0.5661922, -1.2061669, 1.1967747
2: -1.8740723, -1.2480198, -1.8744652, -1.2451730, -0.2128035, 0.2093874
3: -0.3920428, 0.0554515, -0.3921721, 0.0557462, -0.3036781, 0.3039436
4: -0.9812980, 0.1545982, -0.9831635, 0.1602641, -0.9377766, 0.9326121
5: -0.6527903, -0.1999746, -0.6527501, -0.1992413, -0.2448780, 0.2441093
6: -2.7874827, -1.1412506, -2.7881315, -1.1411166, -0.5031492, 0.5038158
7: -0.7870629, 0.3601458, -0.7873808, 0.3628245, -1.0095036, 1.0068314
8: -4.1215158, -0.9956350, -4.1230664, -0.9953654, -1.2500093, 1.2519747
9: -4.9500923, -2.2424142, -4.9499474, -2.2402999, -1.0804141, 1.0638925

Time for backsubstitution: 4.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3097
type: A, layer: 1, pos: 2452
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 334
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 2407
type: A, layer: 1, pos: 337
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 3550
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 364
type: A, layer: 1, pos: 3425
type: A, layer: 1, pos: 350
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 2651
type: A, layer: 1, pos: 2677
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 385
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 261
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 333
type: A, layer: 1, pos: 379
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 396
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 332
type: A, layer: 1, pos: 3549
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 275
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 2584
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 192
type: A, layer: 1, pos: 2108
type: A, layer: 1, pos: 3503
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 3484
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 2109
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 2209
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2970
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 3049
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 395
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3319
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3002
type: A, layer: 1, pos: 3481
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 2217
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2338
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 2312
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 3279
type: A, layer: 1, pos: 3322
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 2113
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 2127
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 2313
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 2106
type: A, layer: 1, pos: 2102
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 2626
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 2086
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 2258
type: A, layer: 1, pos: 2259
type: A, layer: 1, pos: 2446
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 2181
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 2976
type: A, layer: 1, pos: 2206
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 3186
type: A, layer: 1, pos: 2251
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 2250
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2940
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2480
type: A, layer: 1, pos: 2092
type: A, layer: 1, pos: 2595
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 206
type: A, layer: 1, pos: 2068
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2100
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 357
type: A, layer: 1, pos: 2315
type: A, layer: 1, pos: 2507
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2254
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 356
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 3005
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3326
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2260
type: A, layer: 1, pos: 2484
type: A, layer: 1, pos: 2078
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 2118
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 2262
type: A, layer: 1, pos: 2110
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 2263
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 149
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 404
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 2084
type: A, layer: 1, pos: 2099
type: A, layer: 1, pos: 2144
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 2624
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3135
type: A, layer: 1, pos: 3136
type: A, layer: 1, pos: 3142
type: A, layer: 1, pos: 3143
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3149
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3591

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 3097

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2531443, upper bound: 0.2538627
time: 116.93 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2532605, upper bound: 0.2538637
time: 131.45 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -4.3324986, -1.7633498, -4.3321128, -1.7670257, -0.9627413, 0.9668180
1: -3.4987690, -0.5658031, -3.4974148, -0.5655894, -1.1971508, 1.1946216
2: -1.8780509, -1.2430375, -1.8700085, -1.2430885, -0.2188555, 0.2104298
3: -0.3930384, 0.0571225, -0.3928370, 0.0548332, -0.3036130, 0.3058218
4: -0.9841933, 0.1639208, -0.9814116, 0.1545972, -0.9351764, 0.9405313
5: -0.6566537, -0.1972699, -0.6527761, -0.1985059, -0.2492217, 0.2463895
6: -2.7929628, -1.1335874, -2.7921653, -1.1415602, -0.5082266, 0.5164486
7: -0.7909349, 0.3649361, -0.7832246, 0.3646830, -1.0145669, 1.0073715
8: -4.1250467, -0.9942040, -4.1234813, -0.9982710, -1.2521482, 1.2532035
9: -4.9497600, -2.2417645, -4.9483929, -2.2419612, -1.0677691, 1.0637132

Time for backsubstitution: 4.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3097
type: A, layer: 1, pos: 2452
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 334
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 2407
type: A, layer: 1, pos: 337
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 3550
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 261
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 364
type: A, layer: 1, pos: 3425
type: A, layer: 1, pos: 350
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 2651
type: A, layer: 1, pos: 2677
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 385
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 333
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 379
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 396
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 332
type: A, layer: 1, pos: 3549
type: A, layer: 1, pos: 275
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 2584
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 192
type: A, layer: 1, pos: 2108
type: A, layer: 1, pos: 3503
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 3484
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 2109
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 2209
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2970
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 3049
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 395
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3319
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3002
type: A, layer: 1, pos: 3481
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 2217
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 2338
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 2312
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 3279
type: A, layer: 1, pos: 3322
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2113
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 2127
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 2313
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 2106
type: A, layer: 1, pos: 2102
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 2626
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 2086
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 3186
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 2258
type: A, layer: 1, pos: 2259
type: A, layer: 1, pos: 2446
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 2181
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 2976
type: A, layer: 1, pos: 2206
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 2251
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 2250
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2940
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 2480
type: A, layer: 1, pos: 2092
type: A, layer: 1, pos: 2595
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 206
type: A, layer: 1, pos: 2068
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2100
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 357
type: A, layer: 1, pos: 2315
type: A, layer: 1, pos: 2507
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2254
type: A, layer: 1, pos: 356
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3005
type: A, layer: 1, pos: 3326
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2260
type: A, layer: 1, pos: 2484
type: A, layer: 1, pos: 2078
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 2118
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 2262
type: A, layer: 1, pos: 2110
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 2263
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 149
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 404
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 2084
type: A, layer: 1, pos: 2099
type: A, layer: 1, pos: 2144
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 2624
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3135
type: A, layer: 1, pos: 3136
type: A, layer: 1, pos: 3142
type: A, layer: 1, pos: 3143
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3149
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3591

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 3097

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.2537467, upper bound: 0.2533261
time: 70.71 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2538617, upper bound: 0.2533216
time: 538.86 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -4.3330922, -1.7629097, -4.3333387, -1.7664919, -0.9640045, 0.9683917
1: -3.4984977, -0.5658031, -3.4972072, -0.5661888, -1.2067537, 1.1958282
2: -1.8813901, -1.2430389, -1.8745112, -1.2411594, -0.2240753, 0.2139599
3: -0.3932340, 0.0577127, -0.3930730, 0.0558521, -0.3050197, 0.3071190
4: -0.9895518, 0.1639194, -0.9886422, 0.1602612, -0.9462506, 0.9467381
5: -0.6566687, -0.1963303, -0.6527559, -0.1970546, -0.2515201, 0.2479752
6: -2.7929516, -1.1333659, -2.7920442, -1.1411090, -0.5091371, 0.5169405
7: -0.7942297, 0.3649990, -0.7875793, 0.3666246, -1.0199957, 1.0116451
8: -4.1251001, -0.9920459, -4.1247721, -0.9953442, -1.2546487, 1.2569580
9: -4.9507694, -2.2417090, -4.9499435, -2.2398865, -1.0832112, 1.0640023

Time for backsubstitution: 4.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3097
type: A, layer: 1, pos: 2452
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 334
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 2407
type: A, layer: 1, pos: 337
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 3550
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 261
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 364
type: A, layer: 1, pos: 3425
type: A, layer: 1, pos: 350
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 2651
type: A, layer: 1, pos: 2677
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 385
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 333
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 379
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 396
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 332
type: A, layer: 1, pos: 3549
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 275
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 2584
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 192
type: A, layer: 1, pos: 2108
type: A, layer: 1, pos: 3503
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 3484
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 2109
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 2209
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2970
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 3049
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 395
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3319
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3002
type: A, layer: 1, pos: 3481
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 2217
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2338
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 2312
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 3279
type: A, layer: 1, pos: 3322
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2113
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 2127
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 2313
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 2106
type: A, layer: 1, pos: 2102
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 2626
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 2086
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 3186
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 2258
type: A, layer: 1, pos: 2259
type: A, layer: 1, pos: 2446
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 2181
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 2976
type: A, layer: 1, pos: 2206
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 2251
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 2250
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 2940
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 2480
type: A, layer: 1, pos: 2092
type: A, layer: 1, pos: 2595
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 206
type: A, layer: 1, pos: 2068
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2100
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 2315
type: A, layer: 1, pos: 357
type: A, layer: 1, pos: 2507
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2254
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 356
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 3005
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3326
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2260
type: A, layer: 1, pos: 2484
type: A, layer: 1, pos: 2078
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 2118
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 2262
type: A, layer: 1, pos: 2110
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 2263
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 149
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 404
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 2084
type: A, layer: 1, pos: 2099
type: A, layer: 1, pos: 2144
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 2624
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3135
type: A, layer: 1, pos: 3136
type: A, layer: 1, pos: 3142
type: A, layer: 1, pos: 3143
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3149
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3591

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 3097

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2538067, upper bound: 0.2539230
time: 104.76 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2539225, upper bound: 0.2539234
time: 26.12 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 135.33 seconds
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 135.33
Output dim: 4, lower bound: -0.2531443, upper bound: 0.2538627
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 135.33
Output dim: 4, lower bound: -0.2532605, upper bound: 0.2538637
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 135.33
Output dim: 4, lower bound: -0.2537467, upper bound: 0.2533261
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 135.33
Output dim: 4, lower bound: -0.2538617, upper bound: 0.2533216
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 135.33
Output dim: 4, lower bound: -0.2538067, upper bound: 0.2539230
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 135.33
Output dim: 4, lower bound: -0.2539225, upper bound: 0.2539234

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4.3291879, -1.7724330, -4.3311853, -1.7713906, -0.9604495, 0.9585062
1: -3.4979992, -0.5728660, -3.4980407, -0.5723000, -1.2052593, 1.1922436
2: -1.8740332, -1.2485001, -1.8744478, -1.2455806, -0.2127372, 0.2090753
3: -0.3920236, 0.0554027, -0.3921693, 0.0556977, -0.3036268, 0.3038854
4: -0.9764647, 0.1547651, -0.9791709, 0.1602638, -0.9329912, 0.9288498
5: -0.6525880, -0.1998933, -0.6525878, -0.1992493, -0.2447113, 0.2440886
6: -2.7874699, -1.1412487, -2.7881217, -1.1411214, -0.5031085, 0.5037891
7: -0.7866353, 0.3603116, -0.7870286, 0.3628165, -1.0090761, 1.0066435
8: -4.1215010, -1.0037169, -4.1230469, -1.0020902, -1.2487278, 1.2466360
9: -4.9508410, -2.2494991, -4.9499435, -2.2461057, -1.0794616, 1.0589904

Time for backsubstitution: 4.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2452
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 334
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 2407
type: B, layer: 1, pos: 337
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 3550
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 261
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 364
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 3425
type: B, layer: 1, pos: 350
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 2651
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 2231
type: B, layer: 1, pos: 385
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 333
type: B, layer: 1, pos: 379
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 3097
type: B, layer: 1, pos: 332
type: B, layer: 1, pos: 3549
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 275
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 2584
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 2600
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 276
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 192
type: B, layer: 1, pos: 2108
type: B, layer: 1, pos: 3503
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 3484
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 2109
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 2209
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2970
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 3049
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 395
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3319
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3002
type: B, layer: 1, pos: 3481
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 2217
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2338
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2312
type: B, layer: 1, pos: 145
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 3322
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2056
type: B, layer: 1, pos: 2113
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 2127
type: B, layer: 1, pos: 2087
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 2313
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2106
type: B, layer: 1, pos: 2102
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 2626
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 2086
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 3186
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 2258
type: B, layer: 1, pos: 2259
type: B, layer: 1, pos: 2446
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 2181
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 2976
type: B, layer: 1, pos: 2206
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 2251
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 2205
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 2250
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2940
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 2092
type: B, layer: 1, pos: 2595
type: B, layer: 1, pos: 2480
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 206
type: B, layer: 1, pos: 2068
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 2100
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 357
type: B, layer: 1, pos: 2315
type: B, layer: 1, pos: 2507
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2254
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 356
type: B, layer: 1, pos: 2052
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3005
type: B, layer: 1, pos: 3326
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2539
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2260
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 2078
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 2118
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 2051
type: B, layer: 1, pos: 2262
type: B, layer: 1, pos: 2110
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2263
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 149
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 404
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 2084
type: B, layer: 1, pos: 2099
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 2624
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3135
type: B, layer: 1, pos: 3136
type: B, layer: 1, pos: 3142
type: B, layer: 1, pos: 3143
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3149
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3591

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 2452

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.2531192, upper bound: 0.2535918
time: 369.14 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2531210, upper bound: 0.2538378
time: 26.48 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -4.3303375, -1.7675632, -4.3311934, -1.7673424, -0.9608979, 0.9605446
1: -3.4983633, -0.5670626, -3.4980521, -0.5673580, -1.2050540, 1.1952674
2: -1.8740659, -1.2481194, -1.8744595, -1.2452521, -0.2126861, 0.2093107
3: -0.3920420, 0.0554348, -0.3921715, 0.0557329, -0.3036632, 0.3039184
4: -0.9801888, 0.1545982, -0.9822558, 0.1602641, -0.9364926, 0.9316380
5: -0.6527008, -0.1999778, -0.6526791, -0.1992439, -0.2447965, 0.2440717
6: -2.7874787, -1.1412524, -2.7881279, -1.1411184, -0.5031256, 0.5037916
7: -0.7868573, 0.3601395, -0.7872178, 0.3628188, -1.0092920, 1.0066651
8: -4.1215062, -0.9974117, -4.1230583, -0.9967957, -1.2487266, 1.2502298
9: -4.9500890, -2.2442989, -4.9499440, -2.2418797, -1.0790987, 1.0623893

Time for backsubstitution: 4.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2452
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 334
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 2407
type: B, layer: 1, pos: 337
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 3550
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 261
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 364
type: B, layer: 1, pos: 3425
type: B, layer: 1, pos: 350
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 2651
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 2231
type: B, layer: 1, pos: 385
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 333
type: B, layer: 1, pos: 379
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 3097
type: B, layer: 1, pos: 332
type: B, layer: 1, pos: 3549
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 275
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 2584
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 2600
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 276
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 192
type: B, layer: 1, pos: 2108
type: B, layer: 1, pos: 3503
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 3484
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 2109
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 2209
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2970
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 3049
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 395
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3319
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3002
type: B, layer: 1, pos: 3481
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 2217
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2338
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2312
type: B, layer: 1, pos: 145
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 3322
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2056
type: B, layer: 1, pos: 2113
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 2127
type: B, layer: 1, pos: 2087
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 2313
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2106
type: B, layer: 1, pos: 2102
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 2626
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 2086
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 3186
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 2258
type: B, layer: 1, pos: 2259
type: B, layer: 1, pos: 2446
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 2181
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 2976
type: B, layer: 1, pos: 2206
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 2251
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 2205
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 2250
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2940
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 2092
type: B, layer: 1, pos: 2480
type: B, layer: 1, pos: 2595
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 206
type: B, layer: 1, pos: 2068
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 2100
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 357
type: B, layer: 1, pos: 2315
type: B, layer: 1, pos: 2507
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2254
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 356
type: B, layer: 1, pos: 2052
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3005
type: B, layer: 1, pos: 3326
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2539
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2260
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 2078
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 2118
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 2051
type: B, layer: 1, pos: 2262
type: B, layer: 1, pos: 2110
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2263
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 149
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 404
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 2084
type: B, layer: 1, pos: 2099
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 2624
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3135
type: B, layer: 1, pos: 3136
type: B, layer: 1, pos: 3142
type: B, layer: 1, pos: 3143
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3149
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3591

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 2452

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.2532323, upper bound: 0.2535907
time: 562.66 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2532339, upper bound: 0.2538333
time: 544.85 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -4.3324947, -1.7644174, -4.3321104, -1.7678735, -0.9618831, 0.9652252
1: -3.4987597, -0.5672727, -3.4974079, -0.5667570, -1.1960630, 1.1931076
2: -1.8780446, -1.2431371, -1.8700036, -1.2431676, -0.2187418, 0.2103521
3: -0.3930375, 0.0571059, -0.3928363, 0.0548199, -0.3035981, 0.3057970
4: -0.9830853, 0.1639208, -0.9805045, 0.1545974, -0.9338920, 0.9395564
5: -0.6565643, -0.1972732, -0.6527051, -0.1985083, -0.2491405, 0.2463518
6: -2.7929587, -1.1335886, -2.7921615, -1.1415616, -0.5082032, 0.5164242
7: -0.7907296, 0.3649289, -0.7830616, 0.3646780, -1.0143557, 1.0072062
8: -4.1250372, -0.9959824, -4.1234741, -0.9997015, -1.2508651, 1.2514546
9: -4.9497571, -2.2436488, -4.9483900, -2.2435172, -1.0664852, 1.0622183

Time for backsubstitution: 4.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2452
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 334
type: B, layer: 1, pos: 261
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 2407
type: B, layer: 1, pos: 337
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 3550
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 3425
type: B, layer: 1, pos: 364
type: B, layer: 1, pos: 350
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 2651
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 2231
type: B, layer: 1, pos: 385
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 276
type: B, layer: 1, pos: 333
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 379
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 3097
type: B, layer: 1, pos: 332
type: B, layer: 1, pos: 3549
type: B, layer: 1, pos: 275
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 2584
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 2600
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 192
type: B, layer: 1, pos: 3503
type: B, layer: 1, pos: 2108
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 3484
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 2109
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 2209
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2970
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 3049
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 395
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3319
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3002
type: B, layer: 1, pos: 3481
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 2217
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2338
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2312
type: B, layer: 1, pos: 145
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 3322
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2056
type: B, layer: 1, pos: 2113
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 2127
type: B, layer: 1, pos: 2087
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 3186
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 2313
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2106
type: B, layer: 1, pos: 2102
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 2626
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2086
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 2258
type: B, layer: 1, pos: 2259
type: B, layer: 1, pos: 2446
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 2181
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 2976
type: B, layer: 1, pos: 2206
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 2251
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 2205
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 2250
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2940
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 2092
type: B, layer: 1, pos: 2480
type: B, layer: 1, pos: 2595
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 206
type: B, layer: 1, pos: 2068
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 2100
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 357
type: B, layer: 1, pos: 2315
type: B, layer: 1, pos: 2507
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2254
type: B, layer: 1, pos: 356
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 2052
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3005
type: B, layer: 1, pos: 3326
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2539
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2260
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 2078
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 2118
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 2051
type: B, layer: 1, pos: 2262
type: B, layer: 1, pos: 2110
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2263
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 149
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 404
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 2084
type: B, layer: 1, pos: 2099
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 2624
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3135
type: B, layer: 1, pos: 3136
type: B, layer: 1, pos: 3142
type: B, layer: 1, pos: 3143
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3149
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3591

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 2452

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2538337, upper bound: 0.2530521
time: 212.82 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2538351, upper bound: 0.2533004
time: 251.65 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -4.3319387, -1.7688469, -4.3333273, -1.7713866, -0.9626982, 0.9647648
1: -3.4981236, -0.5730748, -3.4971881, -0.5722969, -1.2058461, 1.1912973
2: -1.8813512, -1.2435194, -1.8744943, -1.2415670, -0.2240090, 0.2136479
3: -0.3932149, 0.0576642, -0.3930700, 0.0558037, -0.3049685, 0.3070610
4: -0.9847193, 0.1640865, -0.9846502, 0.1602611, -0.9414672, 0.9429771
5: -0.6564662, -0.1962491, -0.6525937, -0.1970626, -0.2513534, 0.2479539
6: -2.7929385, -1.1333641, -2.7920349, -1.1411135, -0.5090963, 0.5169139
7: -0.7938023, 0.3651648, -0.7872266, 0.3666164, -1.0195677, 1.0114576
8: -4.1250854, -1.0001276, -4.1247525, -1.0020690, -1.2533658, 1.2516189
9: -4.9515185, -2.2487936, -4.9499397, -2.2456925, -1.0822589, 1.0591003

Time for backsubstitution: 4.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2452
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 334
type: B, layer: 1, pos: 261
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 2407
type: B, layer: 1, pos: 337
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 3550
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 364
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 3425
type: B, layer: 1, pos: 350
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 2651
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 2231
type: B, layer: 1, pos: 385
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 276
type: B, layer: 1, pos: 333
type: B, layer: 1, pos: 379
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 3097
type: B, layer: 1, pos: 332
type: B, layer: 1, pos: 3549
type: B, layer: 1, pos: 275
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 2584
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 2600
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 192
type: B, layer: 1, pos: 2108
type: B, layer: 1, pos: 3503
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 3484
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 2109
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 2209
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2970
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 3049
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 395
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3319
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3002
type: B, layer: 1, pos: 3481
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 2217
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2338
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2312
type: B, layer: 1, pos: 145
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 3322
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2056
type: B, layer: 1, pos: 2113
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 2127
type: B, layer: 1, pos: 2087
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 3186
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 2313
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2106
type: B, layer: 1, pos: 2102
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 2626
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 2086
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 2258
type: B, layer: 1, pos: 2259
type: B, layer: 1, pos: 2446
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 2181
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 2976
type: B, layer: 1, pos: 2206
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 2251
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 2205
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 2250
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2940
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 2092
type: B, layer: 1, pos: 2480
type: B, layer: 1, pos: 2595
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 206
type: B, layer: 1, pos: 2068
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 2100
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 357
type: B, layer: 1, pos: 2315
type: B, layer: 1, pos: 2507
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2254
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 2052
type: B, layer: 1, pos: 356
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3005
type: B, layer: 1, pos: 3326
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2539
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2260
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 2078
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 2118
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 2051
type: B, layer: 1, pos: 2262
type: B, layer: 1, pos: 2110
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2263
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 149
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 404
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 2084
type: B, layer: 1, pos: 2099
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 2624
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3135
type: B, layer: 1, pos: 3136
type: B, layer: 1, pos: 3142
type: B, layer: 1, pos: 3143
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3149
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3591

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 2452

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2537827, upper bound: 0.2536534
time: 404.92 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2537812, upper bound: 0.2538979
time: 337.85 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -4.3330894, -1.7639768, -4.3333349, -1.7673389, -0.9631466, 0.9668034
1: -3.4984879, -0.5672727, -3.4971995, -0.5673542, -1.2056408, 1.1943210
2: -1.8813838, -1.2431386, -1.8745059, -1.2412387, -0.2239579, 0.2138832
3: -0.3932333, 0.0576960, -0.3930722, 0.0558389, -0.3050048, 0.3070939
4: -0.9884425, 0.1639195, -0.9877349, 0.1602613, -0.9449670, 0.9457642
5: -0.6565791, -0.1963337, -0.6526849, -0.1970573, -0.2514388, 0.2479374
6: -2.7929471, -1.1333677, -2.7920403, -1.1411099, -0.5091134, 0.5169163
7: -0.7940241, 0.3649922, -0.7874157, 0.3666191, -1.0197839, 1.0114789
8: -4.1250906, -0.9938231, -4.1247644, -0.9967742, -1.2533659, 1.2552133
9: -4.9507670, -2.2435937, -4.9499407, -2.2414665, -1.0818958, 1.0624992

Time for backsubstitution: 4.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2452
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 334
type: B, layer: 1, pos: 261
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 2407
type: B, layer: 1, pos: 337
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 3550
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 364
type: B, layer: 1, pos: 3425
type: B, layer: 1, pos: 350
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 2651
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 2231
type: B, layer: 1, pos: 385
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 276
type: B, layer: 1, pos: 333
type: B, layer: 1, pos: 379
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 3097
type: B, layer: 1, pos: 332
type: B, layer: 1, pos: 3549
type: B, layer: 1, pos: 275
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 2584
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 2600
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 192
type: B, layer: 1, pos: 2108
type: B, layer: 1, pos: 3503
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 3484
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 2109
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 2209
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2970
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 3049
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2280
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 395
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3319
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 3002
type: B, layer: 1, pos: 3481
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 2217
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2338
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 2312
type: B, layer: 1, pos: 145
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 3322
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2056
type: B, layer: 1, pos: 2113
type: B, layer: 1, pos: 2287
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 2127
type: B, layer: 1, pos: 2087
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 3186
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 2313
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2106
type: B, layer: 1, pos: 2102
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 2626
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 2086
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 2258
type: B, layer: 1, pos: 2259
type: B, layer: 1, pos: 2446
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 2181
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 2976
type: B, layer: 1, pos: 2206
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 2251
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 2205
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 3073
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 2250
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2940
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 2092
type: B, layer: 1, pos: 2480
type: B, layer: 1, pos: 2595
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 206
type: B, layer: 1, pos: 2068
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 2100
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 2315
type: B, layer: 1, pos: 357
type: B, layer: 1, pos: 2507
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2254
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 2052
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 356
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3005
type: B, layer: 1, pos: 3326
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2539
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2260
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 2078
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 2118
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 2051
type: B, layer: 1, pos: 2262
type: B, layer: 1, pos: 2110
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2263
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 149
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 404
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 2084
type: B, layer: 1, pos: 2099
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2294
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 2624
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 2969
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3135
type: B, layer: 1, pos: 3136
type: B, layer: 1, pos: 3142
type: B, layer: 1, pos: 3143
type: B, layer: 1, pos: 3144
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3149
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3591

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 2452

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2538957, upper bound: 0.2536543
time: 33.24 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2538967, upper bound: 0.2539005
time: 358.73 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 396.61 seconds
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 396.61
Output dim: 4, lower bound: -0.2531192, upper bound: 0.2535918
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 396.61
Output dim: 4, lower bound: -0.2531210, upper bound: 0.2538378
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 396.61
Output dim: 4, lower bound: -0.2532323, upper bound: 0.2535907
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 396.61
Output dim: 4, lower bound: -0.2532339, upper bound: 0.2538333
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 396.61
Output dim: 4, lower bound: -0.2538337, upper bound: 0.2530521
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 396.61
Output dim: 4, lower bound: -0.2538351, upper bound: 0.2533004
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 396.61
Output dim: 4, lower bound: -0.2537827, upper bound: 0.2536534
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 396.61
Output dim: 4, lower bound: -0.2537812, upper bound: 0.2538979
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 396.61
Output dim: 4, lower bound: -0.2538957, upper bound: 0.2536543
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 396.61
Output dim: 4, lower bound: -0.2538967, upper bound: 0.2539005

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4.3291831, -1.7727959, -4.3311796, -1.7718167, -0.9544966, 0.9584962
1: -3.4979961, -0.5733755, -3.4980359, -0.5728610, -1.1921270, 1.1922379
2: -1.8740242, -1.2485627, -1.8744369, -1.2456561, -0.2122554, 0.2090336
3: -0.3920234, 0.0553920, -0.3921689, 0.0556846, -0.3035989, 0.3038878
4: -0.9761284, 0.1547650, -0.9787996, 0.1602638, -0.9326477, 0.9283149
5: -0.6525322, -0.1998948, -0.6525204, -0.1992513, -0.2446908, 0.2436585
6: -2.7874694, -1.1412592, -2.7881207, -1.1411337, -0.5031127, 0.5037423
7: -0.7865620, 0.3603112, -0.7869390, 0.3628158, -1.0090306, 1.0064350
8: -4.1215005, -1.0043240, -4.1230464, -1.0027621, -1.2321278, 1.2466260
9: -4.9508400, -2.2501588, -4.9499426, -2.2468851, -1.0621769, 1.0589719

Time for backsubstitution: 4.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 334
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 2407
type: A, layer: 1, pos: 337
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 3550
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 364
type: A, layer: 1, pos: 3425
type: A, layer: 1, pos: 350
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 2651
type: A, layer: 1, pos: 2677
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 385
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 261
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 333
type: A, layer: 1, pos: 379
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 396
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 332
type: A, layer: 1, pos: 2452
type: A, layer: 1, pos: 3549
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 275
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 2584
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 192
type: A, layer: 1, pos: 2108
type: A, layer: 1, pos: 3503
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 3484
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 2109
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 2209
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2970
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 3049
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 395
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3319
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3002
type: A, layer: 1, pos: 3481
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 2217
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 2338
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 2312
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 3279
type: A, layer: 1, pos: 3322
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 2113
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 2127
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 2313
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 2106
type: A, layer: 1, pos: 2102
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 2626
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 2086
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 2258
type: A, layer: 1, pos: 2446
type: A, layer: 1, pos: 2259
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 2181
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 2976
type: A, layer: 1, pos: 2206
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 3186
type: A, layer: 1, pos: 2251
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 2250
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2940
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2092
type: A, layer: 1, pos: 2480
type: A, layer: 1, pos: 2595
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 206
type: A, layer: 1, pos: 2068
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2100
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 357
type: A, layer: 1, pos: 2315
type: A, layer: 1, pos: 2507
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2254
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 356
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 3005
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3326
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2260
type: A, layer: 1, pos: 2484
type: A, layer: 1, pos: 2078
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 2118
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 2262
type: A, layer: 1, pos: 2110
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 2263
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 149
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 404
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 2084
type: A, layer: 1, pos: 2099
type: A, layer: 1, pos: 2144
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 2624
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3135
type: A, layer: 1, pos: 3136
type: A, layer: 1, pos: 3142
type: A, layer: 1, pos: 3143
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3149
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3591

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 2197

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2527719, upper bound: 0.2537652
time: 282.10 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2530926, upper bound: 0.2537718
time: 370.41 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4.3303337, -1.7679253, -4.3311872, -1.7677686, -0.9549450, 0.9605346
1: -3.4983602, -0.5675731, -3.4980483, -0.5679183, -1.1919218, 1.1952615
2: -1.8740568, -1.2481818, -1.8744483, -1.2453278, -0.2122043, 0.2092689
3: -0.3920418, 0.0554241, -0.3921712, 0.0557199, -0.3036353, 0.3039209
4: -0.9798527, 0.1545982, -0.9818844, 0.1602639, -0.9361492, 0.9311029
5: -0.6526451, -0.1999794, -0.6526116, -0.1992458, -0.2447760, 0.2436415
6: -2.7874780, -1.1412627, -2.7881265, -1.1411309, -0.5031298, 0.5037448
7: -0.7867842, 0.3601385, -0.7871275, 0.3628185, -1.0092467, 1.0064566
8: -4.1215048, -0.9980190, -4.1230574, -0.9974678, -1.2321265, 1.2502203
9: -4.9500895, -2.2449594, -4.9499435, -2.2426593, -1.0618138, 1.0623709

Time for backsubstitution: 4.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 334
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 2407
type: A, layer: 1, pos: 337
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 3550
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 364
type: A, layer: 1, pos: 3425
type: A, layer: 1, pos: 350
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 2651
type: A, layer: 1, pos: 2677
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 385
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 261
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 333
type: A, layer: 1, pos: 379
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 396
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 332
type: A, layer: 1, pos: 2452
type: A, layer: 1, pos: 3549
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 275
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 2584
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 192
type: A, layer: 1, pos: 2108
type: A, layer: 1, pos: 3503
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 3484
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 2109
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 2209
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2970
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 3049
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2280
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 395
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3319
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 3002
type: A, layer: 1, pos: 3481
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 2217
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 2338
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 2312
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 3279
type: A, layer: 1, pos: 3322
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 2113
type: A, layer: 1, pos: 2287
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 2127
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 2313
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 2106
type: A, layer: 1, pos: 2102
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 2626
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 2086
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 2258
type: A, layer: 1, pos: 2259
type: A, layer: 1, pos: 2446
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 2181
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 2976
type: A, layer: 1, pos: 2206
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 3186
type: A, layer: 1, pos: 2251
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 3073
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 2250
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2940
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2480
type: A, layer: 1, pos: 2092
type: A, layer: 1, pos: 2595
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 206
type: A, layer: 1, pos: 2068
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2100
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 357
type: A, layer: 1, pos: 2315
type: A, layer: 1, pos: 2507
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2254
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 356
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 3005
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3326
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2260
type: A, layer: 1, pos: 2484
type: A, layer: 1, pos: 2078
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 2118
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 2262
type: A, layer: 1, pos: 2110
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 2263
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 149
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 404
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 2084
type: A, layer: 1, pos: 2099
type: A, layer: 1, pos: 2144
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2294
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 2624
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 2969
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3135
type: A, layer: 1, pos: 3136
type: A, layer: 1, pos: 3142
type: A, layer: 1, pos: 3143
type: A, layer: 1, pos: 3144
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3149
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3591

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 2197

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.2527126, upper bound: 0.2531684
time: 402.70 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2531701, upper bound: 0.2537709
time: 175.00 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 582.31 seconds
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 582.31
Output dim: 4, lower bound: -0.2527719, upper bound: 0.2537652
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 582.31
Output dim: 4, lower bound: -0.2530926, upper bound: 0.2537718
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 582.31
Output dim: 4, lower bound: -0.2527126, upper bound: 0.2531684
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 582.31
Output dim: 4, lower bound: -0.2531701, upper bound: 0.2537709
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 582.31
Output dim: 4, lower bound: -0.2538337, upper bound: 0.2530521
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 582.31
Output dim: 4, lower bound: -0.2538351, upper bound: 0.2533004
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 582.31
Output dim: 4, lower bound: -0.2537827, upper bound: 0.2536534
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 582.31
Output dim: 4, lower bound: -0.2537812, upper bound: 0.2538979
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 582.31
Output dim: 4, lower bound: -0.2538957, upper bound: 0.2536543
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 582.31
Output dim: 4, lower bound: -0.2538967, upper bound: 0.2539005

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 342.75 + 7180.32 = 7523.06 seconds
