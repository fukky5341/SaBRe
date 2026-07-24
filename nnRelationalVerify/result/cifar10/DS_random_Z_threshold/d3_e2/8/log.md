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
execution time: IAR + RelationalAnalysis = 5.06 + 336.62 = 341.67 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.2540104, upper bound: 0.2540117

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 2484
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 3063
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2089
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2051
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 3481
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 2251
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2622
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2086
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 2452
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 334
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 3591
type: RSZ, layer: 1, pos: 2068
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 2102
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2052
type: RSZ, layer: 1, pos: 276
type: RSZ, layer: 1, pos: 2329
type: RSZ, layer: 1, pos: 2259
type: RSZ, layer: 1, pos: 2071
type: RSZ, layer: 1, pos: 3049
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 357
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 3002
type: RSZ, layer: 1, pos: 149
type: RSZ, layer: 1, pos: 2090
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 3319
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 2407
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2087
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 2695
type: RSZ, layer: 1, pos: 2100
type: RSZ, layer: 1, pos: 2107
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 404
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 333
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 2507
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 275
type: RSZ, layer: 1, pos: 364
type: RSZ, layer: 1, pos: 3186
type: RSZ, layer: 1, pos: 3097
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 3143
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 2108
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 332
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 3005
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 2209
type: RSZ, layer: 1, pos: 3142
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 2061
type: RSZ, layer: 1, pos: 2446
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 192
type: RSZ, layer: 1, pos: 2181
type: RSZ, layer: 1, pos: 2115
type: RSZ, layer: 1, pos: 2976
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 2651
type: RSZ, layer: 1, pos: 2106
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 2940
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 2044
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 379
type: RSZ, layer: 1, pos: 2205
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2600
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2595
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2262
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 396
type: RSZ, layer: 1, pos: 3549
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2692
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2254
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2127
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 2224
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 125
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 261
type: RSZ, layer: 1, pos: 3326
type: RSZ, layer: 1, pos: 2110
type: RSZ, layer: 1, pos: 2312
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 3018
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 2231
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 395
type: RSZ, layer: 1, pos: 2056
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 385
type: RSZ, layer: 1, pos: 2313
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2584
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 3425
type: RSZ, layer: 1, pos: 2084
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 350
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 2109
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 337
type: RSZ, layer: 1, pos: 2260
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 145
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 2250
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2219
type: RSZ, layer: 1, pos: 356
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 2144
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 2258
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 2648
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 2118
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 3503
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2263
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 644

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3080

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2539826, upper bound: 0.2539733
time: 555.06 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2539768, upper bound: 0.2539853
time: 587.02 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1142.09 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1142.09
Output dim: 4, lower bound: -0.2539826, upper bound: 0.2539733
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1142.09
Output dim: 4, lower bound: -0.2539768, upper bound: 0.2539853

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -4.3332100, -1.7664787, -4.3332100, -1.7664787, -0.9637526, 0.9635215
1: -3.5020158, -0.5655890, -3.5020158, -0.5655890, -1.2013016, 1.2008803
2: -1.8742059, -1.2430845, -1.8742059, -1.2430845, -0.2155562, 0.2155417
3: -0.3931611, 0.0556504, -0.3931611, 0.0556504, -0.3053917, 0.3053879
4: -0.9882242, 0.1546044, -0.9882242, 0.1546044, -0.9380751, 0.9380763
5: -0.6528150, -0.1970142, -0.6528150, -0.1970142, -0.2471724, 0.2471682
6: -2.7923806, -1.1410340, -2.7923806, -1.1410340, -0.5087132, 0.5086979
7: -0.7873561, 0.3647786, -0.7873561, 0.3647786, -1.0118873, 1.0118949
8: -4.1237440, -0.9955738, -4.1237440, -0.9955738, -1.2514361, 1.2510780
9: -4.9525437, -2.2417867, -4.9525437, -2.2417867, -1.0714195, 1.0709119

Time for backsubstitution: 4.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 334
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 2056
type: RSZ, layer: 1, pos: 2181
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2089
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 276
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 379
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 2507
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 337
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 2484
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 3591
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2068
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2071
type: RSZ, layer: 1, pos: 2312
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 2084
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2250
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 364
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 2224
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 2648
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2254
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 2446
type: RSZ, layer: 1, pos: 3549
type: RSZ, layer: 1, pos: 2061
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 2976
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 2052
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 2452
type: RSZ, layer: 1, pos: 2940
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 357
type: RSZ, layer: 1, pos: 332
type: RSZ, layer: 1, pos: 3049
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 333
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 3002
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 2622
type: RSZ, layer: 1, pos: 3319
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 2209
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 2100
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2259
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 2313
type: RSZ, layer: 1, pos: 261
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 2107
type: RSZ, layer: 1, pos: 2108
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 2219
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2600
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 3326
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 2144
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 145
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 2051
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 3481
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 2651
type: RSZ, layer: 1, pos: 2127
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 2329
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 3097
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 125
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2106
type: RSZ, layer: 1, pos: 2251
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2205
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 192
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2263
type: RSZ, layer: 1, pos: 395
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 3425
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 2695
type: RSZ, layer: 1, pos: 2260
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 275
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 3005
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 2087
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 2407
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2110
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 3143
type: RSZ, layer: 1, pos: 350
type: RSZ, layer: 1, pos: 149
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 356
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 3186
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 2044
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2231
type: RSZ, layer: 1, pos: 3142
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2118
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 2595
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 3063
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 2115
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 3018
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2086
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 2102
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 2692
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 396
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 404
type: RSZ, layer: 1, pos: 2109
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2584
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 385
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 3503
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2090
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 2258
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 2262

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2538873, upper bound: 0.2539778
time: 228.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2539836, upper bound: 0.2538841
time: 519.88 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -4.3332100, -1.7664787, -4.3332100, -1.7664787, -0.9635214, 0.9637527
1: -3.5020158, -0.5655890, -3.5020158, -0.5655890, -1.2008804, 1.2013016
2: -1.8742059, -1.2430845, -1.8742059, -1.2430845, -0.2155417, 0.2155562
3: -0.3931611, 0.0556504, -0.3931611, 0.0556504, -0.3053879, 0.3053917
4: -0.9882242, 0.1546044, -0.9882242, 0.1546044, -0.9380761, 0.9380751
5: -0.6528150, -0.1970142, -0.6528150, -0.1970142, -0.2471682, 0.2471724
6: -2.7923806, -1.1410340, -2.7923806, -1.1410340, -0.5086979, 0.5087132
7: -0.7873561, 0.3647786, -0.7873561, 0.3647786, -1.0118949, 1.0118873
8: -4.1237440, -0.9955738, -4.1237440, -0.9955738, -1.2510780, 1.2514362
9: -4.9525437, -2.2417867, -4.9525437, -2.2417867, -1.0709119, 1.0714196

Time for backsubstitution: 4.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3481
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 334
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2144
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 357
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 2051
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 385
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2507
type: RSZ, layer: 1, pos: 2262
type: RSZ, layer: 1, pos: 2106
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 2118
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 192
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2254
type: RSZ, layer: 1, pos: 2600
type: RSZ, layer: 1, pos: 364
type: RSZ, layer: 1, pos: 2484
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2329
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 149
type: RSZ, layer: 1, pos: 333
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 275
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 3049
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2219
type: RSZ, layer: 1, pos: 2068
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2313
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2312
type: RSZ, layer: 1, pos: 2446
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 261
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 2259
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2695
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 2407
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 3549
type: RSZ, layer: 1, pos: 3186
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 2107
type: RSZ, layer: 1, pos: 3591
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 3063
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 3097
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 3005
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 2052
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 2084
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 356
type: RSZ, layer: 1, pos: 276
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 2260
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 3503
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 3002
type: RSZ, layer: 1, pos: 2061
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 2231
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 2089
type: RSZ, layer: 1, pos: 2224
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 337
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2209
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2648
type: RSZ, layer: 1, pos: 2102
type: RSZ, layer: 1, pos: 3425
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 350
type: RSZ, layer: 1, pos: 3143
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 2109
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2086
type: RSZ, layer: 1, pos: 2692
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 2108
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 2263
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 145
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 3142
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 3018
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 3319
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 2110
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2071
type: RSZ, layer: 1, pos: 2115
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 2251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2205
type: RSZ, layer: 1, pos: 2090
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 2258
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2595
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 379
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 2127
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 332
type: RSZ, layer: 1, pos: 2100
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 396
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 3326
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 404
type: RSZ, layer: 1, pos: 2940
type: RSZ, layer: 1, pos: 2056
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 2452
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2181
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 2087
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 2976
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 2044
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2250
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 125
type: RSZ, layer: 1, pos: 2584
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 395
type: RSZ, layer: 1, pos: 2651
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 2622
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3481

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2539743, upper bound: 0.2539582
time: 571.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2539513, upper bound: 0.2539579
time: 407.81 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 983.65 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 983.65
Output dim: 4, lower bound: -0.2538873, upper bound: 0.2539778
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 983.65
Output dim: 4, lower bound: -0.2539836, upper bound: 0.2538841
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 983.65
Output dim: 4, lower bound: -0.2539743, upper bound: 0.2539582
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 983.65
Output dim: 4, lower bound: -0.2539513, upper bound: 0.2539579

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.3332100, -1.7664787, -4.3332100, -1.7664787, -0.9583528, 0.9578937
1: -3.5020158, -0.5655890, -3.5020158, -0.5655890, -1.1948568, 1.1942399
2: -1.8742059, -1.2430845, -1.8742059, -1.2430845, -0.2152257, 0.2151938
3: -0.3931611, 0.0556504, -0.3931611, 0.0556504, -0.3053137, 0.3053095
4: -0.9882242, 0.1546044, -0.9882242, 0.1546044, -0.9379778, 0.9379928
5: -0.6528150, -0.1970142, -0.6528150, -0.1970142, -0.2471905, 0.2471862
6: -2.7923806, -1.1410340, -2.7923806, -1.1410340, -0.5064138, 0.5064920
7: -0.7873561, 0.3647786, -0.7873561, 0.3647786, -1.0119137, 1.0119208
8: -4.1237440, -0.9955738, -4.1237440, -0.9955738, -1.2487537, 1.2483670
9: -4.9525437, -2.2417867, -4.9525437, -2.2417867, -1.0681732, 1.0674944

Time for backsubstitution: 4.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2144
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2484
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2446
type: RSZ, layer: 1, pos: 2219
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 2068
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 404
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 2044
type: RSZ, layer: 1, pos: 2231
type: RSZ, layer: 1, pos: 2056
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 2262
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2258
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 2102
type: RSZ, layer: 1, pos: 2115
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 2100
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 2976
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 379
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2071
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3481
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 3049
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2118
type: RSZ, layer: 1, pos: 3018
type: RSZ, layer: 1, pos: 2452
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2622
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 3005
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2090
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 3326
type: RSZ, layer: 1, pos: 3143
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 3097
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2089
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 2648
type: RSZ, layer: 1, pos: 3503
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 3319
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 334
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 2260
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 2127
type: RSZ, layer: 1, pos: 2086
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 2209
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 2250
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 2407
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 125
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2584
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 3549
type: RSZ, layer: 1, pos: 2600
type: RSZ, layer: 1, pos: 149
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 2224
type: RSZ, layer: 1, pos: 364
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2313
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 2259
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2692
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 332
type: RSZ, layer: 1, pos: 385
type: RSZ, layer: 1, pos: 2051
type: RSZ, layer: 1, pos: 2108
type: RSZ, layer: 1, pos: 350
type: RSZ, layer: 1, pos: 3425
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 145
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 3186
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 2254
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 337
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 2651
type: RSZ, layer: 1, pos: 2110
type: RSZ, layer: 1, pos: 2181
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 3063
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 2061
type: RSZ, layer: 1, pos: 395
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 2695
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 2052
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 2251
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2107
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 2329
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 2263
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 3002
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 2106
type: RSZ, layer: 1, pos: 276
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 2087
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 2109
type: RSZ, layer: 1, pos: 192
type: RSZ, layer: 1, pos: 2312
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 2940
type: RSZ, layer: 1, pos: 3142
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 275
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 2205
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 3591
type: RSZ, layer: 1, pos: 2507
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 357
type: RSZ, layer: 1, pos: 356
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2595
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 261
type: RSZ, layer: 1, pos: 333
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 396
type: RSZ, layer: 1, pos: 2084
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 879

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 923

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2538885, upper bound: 0.2539735
time: 189.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2538837, upper bound: 0.2539725
time: 336.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.3332100, -1.7664787, -4.3332100, -1.7664787, -0.9581250, 0.9581217
1: -3.5020158, -0.5655890, -3.5020158, -0.5655890, -1.1946609, 1.1944356
2: -1.8742059, -1.2430845, -1.8742059, -1.2430845, -0.2152082, 0.2152112
3: -0.3931611, 0.0556504, -0.3931611, 0.0556504, -0.3053133, 0.3053099
4: -0.9882242, 0.1546044, -0.9882242, 0.1546044, -0.9379916, 0.9379790
5: -0.6528150, -0.1970142, -0.6528150, -0.1970142, -0.2471903, 0.2471863
6: -2.7923806, -1.1410340, -2.7923806, -1.1410340, -0.5065073, 0.5063984
7: -0.7873561, 0.3647786, -0.7873561, 0.3647786, -1.0119132, 1.0119213
8: -4.1237440, -0.9955738, -4.1237440, -0.9955738, -1.2487251, 1.2483954
9: -4.9525437, -2.2417867, -4.9525437, -2.2417867, -1.0680020, 1.0676655

Time for backsubstitution: 4.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 337
type: RSZ, layer: 1, pos: 2976
type: RSZ, layer: 1, pos: 3326
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2651
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 2102
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 2258
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 275
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2219
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2061
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2051
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2262
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 379
type: RSZ, layer: 1, pos: 2106
type: RSZ, layer: 1, pos: 2259
type: RSZ, layer: 1, pos: 3097
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 404
type: RSZ, layer: 1, pos: 2089
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2205
type: RSZ, layer: 1, pos: 125
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 2595
type: RSZ, layer: 1, pos: 3319
type: RSZ, layer: 1, pos: 2044
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 334
type: RSZ, layer: 1, pos: 2313
type: RSZ, layer: 1, pos: 364
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 192
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2484
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 2446
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 357
type: RSZ, layer: 1, pos: 2584
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 2251
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 2127
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 2648
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2181
type: RSZ, layer: 1, pos: 3142
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 2068
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 333
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 395
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 3549
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 149
type: RSZ, layer: 1, pos: 2263
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2086
type: RSZ, layer: 1, pos: 3002
type: RSZ, layer: 1, pos: 3481
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 2108
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 3049
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 2260
type: RSZ, layer: 1, pos: 2084
type: RSZ, layer: 1, pos: 385
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 2622
type: RSZ, layer: 1, pos: 2118
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 3063
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 3005
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 3425
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 2452
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 3591
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 350
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 3503
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 2209
type: RSZ, layer: 1, pos: 2224
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 2692
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 2407
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 3186
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 332
type: RSZ, layer: 1, pos: 2329
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 2056
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 2052
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 2090
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2100
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 3018
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 2144
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 2695
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 356
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2231
type: RSZ, layer: 1, pos: 396
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 2250
type: RSZ, layer: 1, pos: 2940
type: RSZ, layer: 1, pos: 2115
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 3143
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2109
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 2110
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2087
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 261
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 2254
type: RSZ, layer: 1, pos: 2507
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 2312
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 145
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2107
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 276
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 2071
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2600
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 673

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2539817, upper bound: 0.2538836
time: 204.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2539817, upper bound: 0.2538843
time: 43.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.3332100, -1.7664787, -4.3332100, -1.7664787, -0.9614309, 0.9617798
1: -3.5020158, -0.5655890, -3.5020158, -0.5655890, -1.1996005, 1.2001500
2: -1.8742059, -1.2430845, -1.8742059, -1.2430845, -0.2151576, 0.2152425
3: -0.3931611, 0.0556504, -0.3931611, 0.0556504, -0.3052422, 0.3052531
4: -0.9882242, 0.1546044, -0.9882242, 0.1546044, -0.9377980, 0.9377699
5: -0.6528150, -0.1970142, -0.6528150, -0.1970142, -0.2467510, 0.2467736
6: -2.7923806, -1.1410340, -2.7923806, -1.1410340, -0.5060848, 0.5059695
7: -0.7873561, 0.3647786, -0.7873561, 0.3647786, -1.0119200, 1.0119113
8: -4.1237440, -0.9955738, -4.1237440, -0.9955738, -1.2509868, 1.2513431
9: -4.9525437, -2.2417867, -4.9525437, -2.2417867, -1.0709083, 1.0714159

Time for backsubstitution: 4.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2484
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3319
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 2051
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2507
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 356
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 2087
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 276
type: RSZ, layer: 1, pos: 2061
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 2089
type: RSZ, layer: 1, pos: 2263
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 261
type: RSZ, layer: 1, pos: 2086
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 2407
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 2090
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2205
type: RSZ, layer: 1, pos: 2144
type: RSZ, layer: 1, pos: 2115
type: RSZ, layer: 1, pos: 3425
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 2181
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 3142
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 2692
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 350
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 3143
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 395
type: RSZ, layer: 1, pos: 2258
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 192
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2127
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2056
type: RSZ, layer: 1, pos: 3326
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 2107
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 2109
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2259
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2648
type: RSZ, layer: 1, pos: 2313
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 2695
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 3549
type: RSZ, layer: 1, pos: 333
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 3186
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 125
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 2231
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 2108
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 2452
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2071
type: RSZ, layer: 1, pos: 379
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2329
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 3097
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 2251
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 275
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 2584
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 2254
type: RSZ, layer: 1, pos: 2044
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2100
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 385
type: RSZ, layer: 1, pos: 149
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 2068
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 357
type: RSZ, layer: 1, pos: 334
type: RSZ, layer: 1, pos: 2052
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 2209
type: RSZ, layer: 1, pos: 2260
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 2622
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 2106
type: RSZ, layer: 1, pos: 2110
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 337
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 3002
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 2651
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 332
type: RSZ, layer: 1, pos: 2940
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 3063
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 3049
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 145
type: RSZ, layer: 1, pos: 2595
type: RSZ, layer: 1, pos: 2446
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 2250
type: RSZ, layer: 1, pos: 404
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 396
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 3591
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 3503
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 364
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 3005
type: RSZ, layer: 1, pos: 2312
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 3018
type: RSZ, layer: 1, pos: 2600
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 2976
type: RSZ, layer: 1, pos: 2084
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2262
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 2118
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 2219
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 2102
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 2224
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 2338

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2095

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2539682, upper bound: 0.2539412
time: 43.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2539608, upper bound: 0.2539499
time: 565.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.3332100, -1.7664787, -4.3332100, -1.7664787, -0.9615487, 0.9616621
1: -3.5020158, -0.5655890, -3.5020158, -0.5655890, -1.1997287, 1.2000215
2: -1.8742059, -1.2430845, -1.8742059, -1.2430845, -0.2152281, 0.2151721
3: -0.3931611, 0.0556504, -0.3931611, 0.0556504, -0.3052493, 0.3052460
4: -0.9882242, 0.1546044, -0.9882242, 0.1546044, -0.9377711, 0.9377968
5: -0.6528150, -0.1970142, -0.6528150, -0.1970142, -0.2467695, 0.2467551
6: -2.7923806, -1.1410340, -2.7923806, -1.1410340, -0.5059543, 0.5061001
7: -0.7873561, 0.3647786, -0.7873561, 0.3647786, -1.0119190, 1.0119122
8: -4.1237440, -0.9955738, -4.1237440, -0.9955738, -1.2509851, 1.2513450
9: -4.9525437, -2.2417867, -4.9525437, -2.2417867, -1.0709082, 1.0714160

Time for backsubstitution: 4.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 2127
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 3142
type: RSZ, layer: 1, pos: 192
type: RSZ, layer: 1, pos: 2259
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 2622
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 350
type: RSZ, layer: 1, pos: 2407
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 2250
type: RSZ, layer: 1, pos: 395
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2106
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 2312
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 149
type: RSZ, layer: 1, pos: 2692
type: RSZ, layer: 1, pos: 2086
type: RSZ, layer: 1, pos: 2595
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2251
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 2329
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 396
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 276
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2056
type: RSZ, layer: 1, pos: 2144
type: RSZ, layer: 1, pos: 357
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 3591
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2313
type: RSZ, layer: 1, pos: 2940
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 2181
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 2263
type: RSZ, layer: 1, pos: 356
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 2584
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2071
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2209
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 2068
type: RSZ, layer: 1, pos: 2052
type: RSZ, layer: 1, pos: 2061
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 334
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 385
type: RSZ, layer: 1, pos: 3186
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 3143
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2118
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 261
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 3425
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 3063
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2651
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 3319
type: RSZ, layer: 1, pos: 3002
type: RSZ, layer: 1, pos: 125
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 2976
type: RSZ, layer: 1, pos: 2087
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 3097
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 2205
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 2051
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 2107
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 2695
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 2115
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 2446
type: RSZ, layer: 1, pos: 2100
type: RSZ, layer: 1, pos: 2090
type: RSZ, layer: 1, pos: 145
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 3503
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 2224
type: RSZ, layer: 1, pos: 3005
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 2484
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 379
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 2260
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 3549
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 2110
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 3326
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2648
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 2109
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 2254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 3018
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2262
type: RSZ, layer: 1, pos: 2089
type: RSZ, layer: 1, pos: 3049
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2102
type: RSZ, layer: 1, pos: 2044
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2084
type: RSZ, layer: 1, pos: 2452
type: RSZ, layer: 1, pos: 2600
type: RSZ, layer: 1, pos: 2108
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 275
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 2231
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 337
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 2507
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 332
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 364
type: RSZ, layer: 1, pos: 404
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 2258
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 333
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 2219
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 671

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 864

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2538315, upper bound: 0.2539793
time: 170.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2539475, upper bound: 0.2538668
time: 302.86 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 478.10 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 478.10
Output dim: 4, lower bound: -0.2538885, upper bound: 0.2539735
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 478.10
Output dim: 4, lower bound: -0.2538837, upper bound: 0.2539725
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 478.10
Output dim: 4, lower bound: -0.2539817, upper bound: 0.2538836
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 478.10
Output dim: 4, lower bound: -0.2539817, upper bound: 0.2538843
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 478.10
Output dim: 4, lower bound: -0.2539682, upper bound: 0.2539412
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 478.10
Output dim: 4, lower bound: -0.2539608, upper bound: 0.2539499
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 478.10
Output dim: 4, lower bound: -0.2538315, upper bound: 0.2539793
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 478.10
Output dim: 4, lower bound: -0.2539475, upper bound: 0.2538668

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.3332100, -1.7664787, -4.3332100, -1.7664787, -0.9583527, 0.9578937
1: -3.5020158, -0.5655890, -3.5020158, -0.5655890, -1.1948566, 1.1942396
2: -1.8742059, -1.2430845, -1.8742059, -1.2430845, -0.2152256, 0.2151937
3: -0.3931611, 0.0556504, -0.3931611, 0.0556504, -0.3053137, 0.3053095
4: -0.9882242, 0.1546044, -0.9882242, 0.1546044, -0.9379778, 0.9379928
5: -0.6528150, -0.1970142, -0.6528150, -0.1970142, -0.2471905, 0.2471862
6: -2.7923806, -1.1410340, -2.7923806, -1.1410340, -0.5064137, 0.5064919
7: -0.7873561, 0.3647786, -0.7873561, 0.3647786, -1.0119135, 1.0119208
8: -4.1237440, -0.9955738, -4.1237440, -0.9955738, -1.2487537, 1.2483670
9: -4.9525437, -2.2417867, -4.9525437, -2.2417867, -1.0681729, 1.0674942

Time for backsubstitution: 4.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2254
type: RSZ, layer: 1, pos: 2940
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 3503
type: RSZ, layer: 1, pos: 2600
type: RSZ, layer: 1, pos: 2250
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 2108
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 2084
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 2205
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 3018
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 2452
type: RSZ, layer: 1, pos: 2068
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 2056
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 261
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2100
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 276
type: RSZ, layer: 1, pos: 2507
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 2260
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 2181
type: RSZ, layer: 1, pos: 2446
type: RSZ, layer: 1, pos: 3326
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2263
type: RSZ, layer: 1, pos: 3143
type: RSZ, layer: 1, pos: 3591
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 2086
type: RSZ, layer: 1, pos: 2312
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2127
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 2087
type: RSZ, layer: 1, pos: 364
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 2484
type: RSZ, layer: 1, pos: 2329
type: RSZ, layer: 1, pos: 3063
type: RSZ, layer: 1, pos: 379
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 125
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2090
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 2071
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 3186
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 2622
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 334
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 3097
type: RSZ, layer: 1, pos: 2107
type: RSZ, layer: 1, pos: 2224
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 332
type: RSZ, layer: 1, pos: 2231
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2106
type: RSZ, layer: 1, pos: 2313
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 356
type: RSZ, layer: 1, pos: 2051
type: RSZ, layer: 1, pos: 3319
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 2115
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 3425
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 357
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 350
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2251
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 3049
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 2052
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 2695
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 3481
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2044
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 2262
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 333
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 149
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2651
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2648
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 2407
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 2595
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 2976
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2109
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 2102
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 3549
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 3142
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 2089
type: RSZ, layer: 1, pos: 192
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 2110
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2144
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2061
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2259
type: RSZ, layer: 1, pos: 2692
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 145
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 3005
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 2209
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 337
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2118
type: RSZ, layer: 1, pos: 396
type: RSZ, layer: 1, pos: 404
type: RSZ, layer: 1, pos: 385
type: RSZ, layer: 1, pos: 395
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 2219
type: RSZ, layer: 1, pos: 2258
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 275
type: RSZ, layer: 1, pos: 2584
type: RSZ, layer: 1, pos: 3002

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3148

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2538771, upper bound: 0.2539745
time: 349.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2538771, upper bound: 0.2539722
time: 185.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.3332100, -1.7664787, -4.3332100, -1.7664787, -0.9583527, 0.9578937
1: -3.5020158, -0.5655890, -3.5020158, -0.5655890, -1.1948566, 1.1942396
2: -1.8742059, -1.2430845, -1.8742059, -1.2430845, -0.2152256, 0.2151937
3: -0.3931611, 0.0556504, -0.3931611, 0.0556504, -0.3053137, 0.3053095
4: -0.9882242, 0.1546044, -0.9882242, 0.1546044, -0.9379778, 0.9379928
5: -0.6528150, -0.1970142, -0.6528150, -0.1970142, -0.2471905, 0.2471862
6: -2.7923806, -1.1410340, -2.7923806, -1.1410340, -0.5064137, 0.5064919
7: -0.7873561, 0.3647786, -0.7873561, 0.3647786, -1.0119135, 1.0119208
8: -4.1237440, -0.9955738, -4.1237440, -0.9955738, -1.2487537, 1.2483670
9: -4.9525437, -2.2417867, -4.9525437, -2.2417867, -1.0681729, 1.0674943

Time for backsubstitution: 4.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2976
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 2052
type: RSZ, layer: 1, pos: 2312
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 3142
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2181
type: RSZ, layer: 1, pos: 3049
type: RSZ, layer: 1, pos: 332
type: RSZ, layer: 1, pos: 2118
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 357
type: RSZ, layer: 1, pos: 2108
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 3002
type: RSZ, layer: 1, pos: 2106
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 3591
type: RSZ, layer: 1, pos: 2484
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 2258
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 3425
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2089
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2648
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 334
type: RSZ, layer: 1, pos: 2044
type: RSZ, layer: 1, pos: 3481
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 3319
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 192
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 2102
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 350
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 3186
type: RSZ, layer: 1, pos: 2940
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 337
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2205
type: RSZ, layer: 1, pos: 2231
type: RSZ, layer: 1, pos: 3549
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 2263
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 2068
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2313
type: RSZ, layer: 1, pos: 2086
type: RSZ, layer: 1, pos: 2622
type: RSZ, layer: 1, pos: 2107
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 2209
type: RSZ, layer: 1, pos: 2127
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2071
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 2259
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 3097
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 261
type: RSZ, layer: 1, pos: 2692
type: RSZ, layer: 1, pos: 3063
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 3503
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 2695
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 2251
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 2329
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 333
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2507
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 2407
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 2595
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 2452
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 2254
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 385
type: RSZ, layer: 1, pos: 275
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 2600
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 356
type: RSZ, layer: 1, pos: 404
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 2250
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 145
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 2084
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 2061
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2446
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 2651
type: RSZ, layer: 1, pos: 396
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 2584
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2262
type: RSZ, layer: 1, pos: 276
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2087
type: RSZ, layer: 1, pos: 2115
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 149
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 364
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 3326
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 2224
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 2110
type: RSZ, layer: 1, pos: 2100
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2219
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 2109
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 395
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 2056
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 3005
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 3018
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 2090
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 125
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 379
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 3143
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 2051
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 2260
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 843

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3367

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2538764, upper bound: 0.2539679
time: 254.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2538764, upper bound: 0.2539704
time: 45.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.3332100, -1.7664787, -4.3332100, -1.7664787, -0.9581250, 0.9581217
1: -3.5020158, -0.5655890, -3.5020158, -0.5655890, -1.1946609, 1.1944356
2: -1.8742059, -1.2430845, -1.8742059, -1.2430845, -0.2152082, 0.2152112
3: -0.3931611, 0.0556504, -0.3931611, 0.0556504, -0.3053133, 0.3053099
4: -0.9882242, 0.1546044, -0.9882242, 0.1546044, -0.9379916, 0.9379790
5: -0.6528150, -0.1970142, -0.6528150, -0.1970142, -0.2471903, 0.2471863
6: -2.7923806, -1.1410340, -2.7923806, -1.1410340, -0.5065073, 0.5063984
7: -0.7873561, 0.3647786, -0.7873561, 0.3647786, -1.0119132, 1.0119213
8: -4.1237440, -0.9955738, -4.1237440, -0.9955738, -1.2487251, 1.2483954
9: -4.9525437, -2.2417867, -4.9525437, -2.2417867, -1.0680020, 1.0676655

Time for backsubstitution: 4.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 404
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2205
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 2600
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2102
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 2224
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 332
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 3481
type: RSZ, layer: 1, pos: 3549
type: RSZ, layer: 1, pos: 2087
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2254
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 2695
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2110
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 2090
type: RSZ, layer: 1, pos: 2313
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 2086
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2144
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2259
type: RSZ, layer: 1, pos: 2260
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2648
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 3018
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 2115
type: RSZ, layer: 1, pos: 2622
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 3425
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2312
type: RSZ, layer: 1, pos: 3326
type: RSZ, layer: 1, pos: 2044
type: RSZ, layer: 1, pos: 357
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 356
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 2051
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2976
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2068
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 2089
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2940
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 2118
type: RSZ, layer: 1, pos: 364
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 3005
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2584
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 396
type: RSZ, layer: 1, pos: 395
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 2061
type: RSZ, layer: 1, pos: 261
type: RSZ, layer: 1, pos: 2595
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 337
type: RSZ, layer: 1, pos: 2484
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2071
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 2258
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 2209
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 3097
type: RSZ, layer: 1, pos: 2329
type: RSZ, layer: 1, pos: 2692
type: RSZ, layer: 1, pos: 2052
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2127
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 2250
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 333
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 125
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 2109
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2056
type: RSZ, layer: 1, pos: 2108
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 3002
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2507
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 3049
type: RSZ, layer: 1, pos: 3186
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 275
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 2263
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 379
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 2100
type: RSZ, layer: 1, pos: 334
type: RSZ, layer: 1, pos: 2106
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2181
type: RSZ, layer: 1, pos: 3063
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 3591
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 2084
type: RSZ, layer: 1, pos: 192
type: RSZ, layer: 1, pos: 3143
type: RSZ, layer: 1, pos: 276
type: RSZ, layer: 1, pos: 3503
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 145
type: RSZ, layer: 1, pos: 3319
type: RSZ, layer: 1, pos: 385
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 2452
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2107
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 2251
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 149
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2446
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 3142
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 2651
type: RSZ, layer: 1, pos: 350
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 2407
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 2219
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2231
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2262
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 158

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 404

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2539772, upper bound: 0.2538762
time: 304.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2539772, upper bound: 0.2538833
time: 350.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.3332100, -1.7664787, -4.3332100, -1.7664787, -0.9581250, 0.9581217
1: -3.5020158, -0.5655890, -3.5020158, -0.5655890, -1.1946609, 1.1944356
2: -1.8742059, -1.2430845, -1.8742059, -1.2430845, -0.2152082, 0.2152112
3: -0.3931611, 0.0556504, -0.3931611, 0.0556504, -0.3053133, 0.3053099
4: -0.9882242, 0.1546044, -0.9882242, 0.1546044, -0.9379916, 0.9379790
5: -0.6528150, -0.1970142, -0.6528150, -0.1970142, -0.2471903, 0.2471863
6: -2.7923806, -1.1410340, -2.7923806, -1.1410340, -0.5065073, 0.5063984
7: -0.7873561, 0.3647786, -0.7873561, 0.3647786, -1.0119132, 1.0119213
8: -4.1237440, -0.9955738, -4.1237440, -0.9955738, -1.2487251, 1.2483954
9: -4.9525437, -2.2417867, -4.9525437, -2.2417867, -1.0680020, 1.0676655

Time for backsubstitution: 4.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 125
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 2110
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 2648
type: RSZ, layer: 1, pos: 2056
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 149
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 332
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 2061
type: RSZ, layer: 1, pos: 2209
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 3143
type: RSZ, layer: 1, pos: 3425
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 3097
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 2108
type: RSZ, layer: 1, pos: 192
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2231
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 2976
type: RSZ, layer: 1, pos: 3002
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 2258
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 2115
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 2622
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 2600
type: RSZ, layer: 1, pos: 2940
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 2052
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 2089
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 396
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 2086
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 2313
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 2109
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2127
type: RSZ, layer: 1, pos: 364
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 2100
type: RSZ, layer: 1, pos: 2584
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 3005
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 145
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 2692
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 2051
type: RSZ, layer: 1, pos: 3186
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 2312
type: RSZ, layer: 1, pos: 2044
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 3049
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 2263
type: RSZ, layer: 1, pos: 2262
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 356
type: RSZ, layer: 1, pos: 2106
type: RSZ, layer: 1, pos: 2484
type: RSZ, layer: 1, pos: 3481
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 2695
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 3591
type: RSZ, layer: 1, pos: 2507
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 3326
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 2087
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2329
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 261
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 385
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 2260
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2084
type: RSZ, layer: 1, pos: 395
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 3503
type: RSZ, layer: 1, pos: 2118
type: RSZ, layer: 1, pos: 2107
type: RSZ, layer: 1, pos: 334
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 2205
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 2651
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 2595
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2068
type: RSZ, layer: 1, pos: 357
type: RSZ, layer: 1, pos: 3063
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 275
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 2251
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2144
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2452
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 337
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 2224
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 276
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2446
type: RSZ, layer: 1, pos: 2259
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 2219
type: RSZ, layer: 1, pos: 2250
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 3319
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 404
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 3018
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 3142
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 2407
type: RSZ, layer: 1, pos: 2071
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 2090
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 2254
type: RSZ, layer: 1, pos: 379
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 333
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 2102
type: RSZ, layer: 1, pos: 2181
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 3549
type: RSZ, layer: 1, pos: 350

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2405

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2539326, upper bound: 0.2538593
time: 313.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2539597, upper bound: 0.2538310
time: 435.55 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 753.59 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 753.59
Output dim: 4, lower bound: -0.2538771, upper bound: 0.2539745
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 753.59
Output dim: 4, lower bound: -0.2538771, upper bound: 0.2539722
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 753.59
Output dim: 4, lower bound: -0.2538764, upper bound: 0.2539679
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 753.59
Output dim: 4, lower bound: -0.2538764, upper bound: 0.2539704
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 753.59
Output dim: 4, lower bound: -0.2539772, upper bound: 0.2538762
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 753.59
Output dim: 4, lower bound: -0.2539772, upper bound: 0.2538833
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 753.59
Output dim: 4, lower bound: -0.2539326, upper bound: 0.2538593
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 753.59
Output dim: 4, lower bound: -0.2539597, upper bound: 0.2538310
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 753.59
Output dim: 4, lower bound: -0.2539682, upper bound: 0.2539412
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 753.59
Output dim: 4, lower bound: -0.2539608, upper bound: 0.2539499
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 753.59
Output dim: 4, lower bound: -0.2538315, upper bound: 0.2539793
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 753.59
Output dim: 4, lower bound: -0.2539475, upper bound: 0.2538668

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 341.67 + 7009.34 = 7351.01 seconds
