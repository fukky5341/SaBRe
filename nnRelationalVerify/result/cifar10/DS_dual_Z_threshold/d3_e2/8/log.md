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
0: (-4.3332100, -1.7664790, -4.3332100, -1.7664790, -0.9647808, 0.9647808)
1: (-3.5020158, -0.5655885, -3.5020158, -0.5655885, -1.2024612, 1.2024612)
2: (-1.8742058, -1.2430847, -1.8742058, -1.2430847, -0.2156220, 0.2156219)
3: (-0.3931611, 0.0556504, -0.3931611, 0.0556504, -0.3053983, 0.3053983)
4: (-0.9882243, 0.1546044, -0.9882243, 0.1546044, -0.9380938, 0.9380937)
5: (-0.6528150, -0.1970140, -0.6528150, -0.1970140, -0.2471912, 0.2471912)
6: (-2.7923801, -1.1410341, -2.7923801, -1.1410341, -0.5087214, 0.5087214)
7: (-0.7873561, 0.3647786, -0.7873561, 0.3647786, -1.0119059, 1.0119059)
8: (-4.1237435, -0.9955735, -4.1237435, -0.9955735, -1.2521670, 1.2521672)
9: (-4.9525442, -2.2417877, -4.9525442, -2.2417877, -1.0728310, 1.0728310)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 5.41 + 266.18 = 271.60 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.2540000, upper bound: 0.2540122

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 276
type: RSZ, layer: 1, pos: 261
type: RSZ, layer: 1, pos: 379
type: RSZ, layer: 1, pos: 275
type: RSZ, layer: 1, pos: 3002
type: RSZ, layer: 1, pos: 364
type: RSZ, layer: 1, pos: 2312
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2313
type: RSZ, layer: 1, pos: 2102
type: RSZ, layer: 1, pos: 2087
type: RSZ, layer: 1, pos: 3018
type: RSZ, layer: 1, pos: 3005
type: RSZ, layer: 1, pos: 2086
type: RSZ, layer: 1, pos: 2329
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 2100
type: RSZ, layer: 1, pos: 2446
type: RSZ, layer: 1, pos: 2507
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 395
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2089
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3186
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 2090
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2205
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 2071
type: RSZ, layer: 1, pos: 2118
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 2056
type: RSZ, layer: 1, pos: 2115
type: RSZ, layer: 1, pos: 3319
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 2106
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2976
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 2061
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2107
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2584
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 2181
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 3049
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2600
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 396
type: RSZ, layer: 1, pos: 125
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 333
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 332
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 2407
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 3063
type: RSZ, layer: 1, pos: 3425
type: RSZ, layer: 1, pos: 2595
type: RSZ, layer: 1, pos: 3097
type: RSZ, layer: 1, pos: 334
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 2044
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 3481
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2484
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 2258
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2254
type: RSZ, layer: 1, pos: 2259
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 350
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 145
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 149
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 192
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 337
type: RSZ, layer: 1, pos: 356
type: RSZ, layer: 1, pos: 357
type: RSZ, layer: 1, pos: 385
type: RSZ, layer: 1, pos: 404
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 2051
type: RSZ, layer: 1, pos: 2052
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 2068
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2084
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2108
type: RSZ, layer: 1, pos: 2109
type: RSZ, layer: 1, pos: 2110
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2127
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2144
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2209
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 2219
type: RSZ, layer: 1, pos: 2224
type: RSZ, layer: 1, pos: 2231
type: RSZ, layer: 1, pos: 2250
type: RSZ, layer: 1, pos: 2251
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2260
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2262
type: RSZ, layer: 1, pos: 2263
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 2452
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2622
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 2648
type: RSZ, layer: 1, pos: 2651
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 2692
type: RSZ, layer: 1, pos: 2695
type: RSZ, layer: 1, pos: 2940
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 3142
type: RSZ, layer: 1, pos: 3143
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 3326
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3503
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 3549
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 3591

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 1, pos: 276

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2533896, upper bound: 0.2540101
time: 44.06 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2539944, upper bound: 0.2534034
time: 198.06 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 242.55 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 242.55
Output dim: 4, lower bound: -0.2533896, upper bound: 0.2540101
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 242.55
Output dim: 4, lower bound: -0.2539944, upper bound: 0.2534034

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -4.3332100, -1.7664790, -4.3332100, -1.7664790, -0.9642710, 0.9643113
1: -3.5020158, -0.5655885, -3.5020158, -0.5655885, -1.2024199, 1.2024062
2: -1.8742058, -1.2430847, -1.8742058, -1.2430847, -0.2152097, 0.2151521
3: -0.3931611, 0.0556504, -0.3931611, 0.0556504, -0.3053622, 0.3053620
4: -0.9882243, 0.1546044, -0.9882243, 0.1546044, -0.9380842, 0.9380884
5: -0.6528150, -0.1970140, -0.6528150, -0.1970140, -0.2469908, 0.2469853
6: -2.7923801, -1.1410341, -2.7923801, -1.1410341, -0.5087218, 0.5087211
7: -0.7873561, 0.3647786, -0.7873561, 0.3647786, -1.0115485, 1.0115466
8: -4.1237435, -0.9955735, -4.1237435, -0.9955735, -1.2518277, 1.2518570
9: -4.9525442, -2.2417877, -4.9525442, -2.2417877, -1.0726928, 1.0726655

Time for backsubstitution: 2.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 261
type: RSZ, layer: 1, pos: 379
type: RSZ, layer: 1, pos: 275
type: RSZ, layer: 1, pos: 3002
type: RSZ, layer: 1, pos: 364
type: RSZ, layer: 1, pos: 2312
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2313
type: RSZ, layer: 1, pos: 2102
type: RSZ, layer: 1, pos: 2087
type: RSZ, layer: 1, pos: 3018
type: RSZ, layer: 1, pos: 3005
type: RSZ, layer: 1, pos: 2086
type: RSZ, layer: 1, pos: 2329
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 2100
type: RSZ, layer: 1, pos: 2446
type: RSZ, layer: 1, pos: 2507
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 395
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2089
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3186
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 2090
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2205
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 2071
type: RSZ, layer: 1, pos: 2118
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 2056
type: RSZ, layer: 1, pos: 2115
type: RSZ, layer: 1, pos: 3319
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 2106
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2976
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 2061
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2107
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2584
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 2181
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 3049
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2600
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 396
type: RSZ, layer: 1, pos: 125
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 333
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 332
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 2407
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 3063
type: RSZ, layer: 1, pos: 3425
type: RSZ, layer: 1, pos: 2595
type: RSZ, layer: 1, pos: 3097
type: RSZ, layer: 1, pos: 334
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 2044
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 3481
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2484
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 2258
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2254
type: RSZ, layer: 1, pos: 2259
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 350
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 145
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 149
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 192
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 337
type: RSZ, layer: 1, pos: 356
type: RSZ, layer: 1, pos: 357
type: RSZ, layer: 1, pos: 385
type: RSZ, layer: 1, pos: 404
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 2051
type: RSZ, layer: 1, pos: 2052
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 2068
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2084
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2108
type: RSZ, layer: 1, pos: 2109
type: RSZ, layer: 1, pos: 2110
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2127
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2144
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2209
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 2219
type: RSZ, layer: 1, pos: 2224
type: RSZ, layer: 1, pos: 2231
type: RSZ, layer: 1, pos: 2250
type: RSZ, layer: 1, pos: 2251
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2260
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2262
type: RSZ, layer: 1, pos: 2263
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 2452
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2622
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 2648
type: RSZ, layer: 1, pos: 2651
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 2692
type: RSZ, layer: 1, pos: 2695
type: RSZ, layer: 1, pos: 2940
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 3142
type: RSZ, layer: 1, pos: 3143
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 3326
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3503
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 3549
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 3591

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 261

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2530252, upper bound: 0.2540051
time: 341.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.2533591, upper bound: 0.2535706
time: 29.08 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -4.3332100, -1.7664790, -4.3332100, -1.7664790, -0.9643114, 0.9642710
1: -3.5020158, -0.5655885, -3.5020158, -0.5655885, -1.2024062, 1.2024199
2: -1.8742058, -1.2430847, -1.8742058, -1.2430847, -0.2151521, 0.2152097
3: -0.3931611, 0.0556504, -0.3931611, 0.0556504, -0.3053620, 0.3053622
4: -0.9882243, 0.1546044, -0.9882243, 0.1546044, -0.9380884, 0.9380843
5: -0.6528150, -0.1970140, -0.6528150, -0.1970140, -0.2469853, 0.2469908
6: -2.7923801, -1.1410341, -2.7923801, -1.1410341, -0.5087211, 0.5087218
7: -0.7873561, 0.3647786, -0.7873561, 0.3647786, -1.0115467, 1.0115485
8: -4.1237435, -0.9955735, -4.1237435, -0.9955735, -1.2518568, 1.2518278
9: -4.9525442, -2.2417877, -4.9525442, -2.2417877, -1.0726655, 1.0726929

Time for backsubstitution: 2.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 261
type: RSZ, layer: 1, pos: 379
type: RSZ, layer: 1, pos: 275
type: RSZ, layer: 1, pos: 3002
type: RSZ, layer: 1, pos: 364
type: RSZ, layer: 1, pos: 2312
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2313
type: RSZ, layer: 1, pos: 2102
type: RSZ, layer: 1, pos: 2087
type: RSZ, layer: 1, pos: 3018
type: RSZ, layer: 1, pos: 3005
type: RSZ, layer: 1, pos: 2086
type: RSZ, layer: 1, pos: 2329
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 2100
type: RSZ, layer: 1, pos: 2446
type: RSZ, layer: 1, pos: 2507
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 395
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2089
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3186
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 2090
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2205
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 2071
type: RSZ, layer: 1, pos: 2118
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 2056
type: RSZ, layer: 1, pos: 2115
type: RSZ, layer: 1, pos: 3319
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 2106
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2976
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 2061
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2107
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2584
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 2181
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 3049
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2600
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 396
type: RSZ, layer: 1, pos: 125
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 333
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 332
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 2407
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 3063
type: RSZ, layer: 1, pos: 3425
type: RSZ, layer: 1, pos: 2595
type: RSZ, layer: 1, pos: 3097
type: RSZ, layer: 1, pos: 334
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 2044
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 3481
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2484
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 2258
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2254
type: RSZ, layer: 1, pos: 2259
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 350
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 145
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 149
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 192
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 337
type: RSZ, layer: 1, pos: 356
type: RSZ, layer: 1, pos: 357
type: RSZ, layer: 1, pos: 385
type: RSZ, layer: 1, pos: 404
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 2051
type: RSZ, layer: 1, pos: 2052
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 2068
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2084
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2108
type: RSZ, layer: 1, pos: 2109
type: RSZ, layer: 1, pos: 2110
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2127
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2144
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2209
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 2219
type: RSZ, layer: 1, pos: 2224
type: RSZ, layer: 1, pos: 2231
type: RSZ, layer: 1, pos: 2250
type: RSZ, layer: 1, pos: 2251
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2260
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2262
type: RSZ, layer: 1, pos: 2263
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 2452
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2622
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 2648
type: RSZ, layer: 1, pos: 2651
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 2692
type: RSZ, layer: 1, pos: 2695
type: RSZ, layer: 1, pos: 2940
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 3142
type: RSZ, layer: 1, pos: 3143
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 3326
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3503
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 3549
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 3591

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 261

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.2535599, upper bound: 0.2533727
time: 35.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2539937, upper bound: 0.2530382
time: 338.00 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 376.29 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 376.29
Output dim: 4, lower bound: -0.2530252, upper bound: 0.2540051
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 376.29
Output dim: 4, lower bound: -0.2533591, upper bound: 0.2535706
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 376.29
Output dim: 4, lower bound: -0.2535599, upper bound: 0.2533727
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 376.29
Output dim: 4, lower bound: -0.2539937, upper bound: 0.2530382

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.3332100, -1.7664790, -4.3332100, -1.7664790, -0.9641895, 0.9642138
1: -3.5020158, -0.5655885, -3.5020158, -0.5655885, -1.2023538, 1.2022876
2: -1.8742058, -1.2430847, -1.8742058, -1.2430847, -0.2150591, 0.2149481
3: -0.3931611, 0.0556504, -0.3931611, 0.0556504, -0.3053617, 0.3053610
4: -0.9882243, 0.1546044, -0.9882243, 0.1546044, -0.9379785, 0.9380117
5: -0.6528150, -0.1970140, -0.6528150, -0.1970140, -0.2469818, 0.2469719
6: -2.7923801, -1.1410341, -2.7923801, -1.1410341, -0.5086415, 0.5086555
7: -0.7873561, 0.3647786, -0.7873561, 0.3647786, -1.0115397, 1.0115297
8: -4.1237435, -0.9955735, -4.1237435, -0.9955735, -1.2518274, 1.2518566
9: -4.9525442, -2.2417877, -4.9525442, -2.2417877, -1.0726877, 1.0726573

Time for backsubstitution: 2.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 379
type: RSZ, layer: 1, pos: 275
type: RSZ, layer: 1, pos: 3002
type: RSZ, layer: 1, pos: 364
type: RSZ, layer: 1, pos: 2312
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2313
type: RSZ, layer: 1, pos: 2102
type: RSZ, layer: 1, pos: 2087
type: RSZ, layer: 1, pos: 3018
type: RSZ, layer: 1, pos: 3005
type: RSZ, layer: 1, pos: 2086
type: RSZ, layer: 1, pos: 2329
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 2100
type: RSZ, layer: 1, pos: 2446
type: RSZ, layer: 1, pos: 2507
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 395
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2089
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3186
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 2090
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2205
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 2071
type: RSZ, layer: 1, pos: 2118
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 2056
type: RSZ, layer: 1, pos: 2115
type: RSZ, layer: 1, pos: 3319
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 2106
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2976
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 2061
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2107
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2584
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 2181
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 3049
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2600
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 396
type: RSZ, layer: 1, pos: 125
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 333
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 332
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 2407
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 3063
type: RSZ, layer: 1, pos: 3425
type: RSZ, layer: 1, pos: 2595
type: RSZ, layer: 1, pos: 3097
type: RSZ, layer: 1, pos: 334
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 2044
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 3481
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2484
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 2258
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2254
type: RSZ, layer: 1, pos: 2259
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 350
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 145
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 149
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 192
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 337
type: RSZ, layer: 1, pos: 356
type: RSZ, layer: 1, pos: 357
type: RSZ, layer: 1, pos: 385
type: RSZ, layer: 1, pos: 404
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 2051
type: RSZ, layer: 1, pos: 2052
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 2068
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2084
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2108
type: RSZ, layer: 1, pos: 2109
type: RSZ, layer: 1, pos: 2110
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2127
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2144
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2209
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 2219
type: RSZ, layer: 1, pos: 2224
type: RSZ, layer: 1, pos: 2231
type: RSZ, layer: 1, pos: 2250
type: RSZ, layer: 1, pos: 2251
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2260
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2262
type: RSZ, layer: 1, pos: 2263
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 2452
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2622
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 2648
type: RSZ, layer: 1, pos: 2651
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 2692
type: RSZ, layer: 1, pos: 2695
type: RSZ, layer: 1, pos: 2940
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 3142
type: RSZ, layer: 1, pos: 3143
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 3326
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3503
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 3549
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 3591

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 379

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2529597, upper bound: 0.2539768
time: 173.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2529960, upper bound: 0.2539373
time: 195.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.3332100, -1.7664790, -4.3332100, -1.7664790, -0.9642137, 0.9641895
1: -3.5020158, -0.5655885, -3.5020158, -0.5655885, -1.2022874, 1.2023541
2: -1.8742058, -1.2430847, -1.8742058, -1.2430847, -0.2149481, 0.2150590
3: -0.3931611, 0.0556504, -0.3931611, 0.0556504, -0.3053611, 0.3053616
4: -0.9882243, 0.1546044, -0.9882243, 0.1546044, -0.9380116, 0.9379785
5: -0.6528150, -0.1970140, -0.6528150, -0.1970140, -0.2469719, 0.2469818
6: -2.7923801, -1.1410341, -2.7923801, -1.1410341, -0.5086555, 0.5086414
7: -0.7873561, 0.3647786, -0.7873561, 0.3647786, -1.0115297, 1.0115397
8: -4.1237435, -0.9955735, -4.1237435, -0.9955735, -1.2518563, 1.2518275
9: -4.9525442, -2.2417877, -4.9525442, -2.2417877, -1.0726572, 1.0726877

Time for backsubstitution: 2.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 379
type: RSZ, layer: 1, pos: 275
type: RSZ, layer: 1, pos: 3002
type: RSZ, layer: 1, pos: 364
type: RSZ, layer: 1, pos: 2312
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2313
type: RSZ, layer: 1, pos: 2102
type: RSZ, layer: 1, pos: 2087
type: RSZ, layer: 1, pos: 3018
type: RSZ, layer: 1, pos: 3005
type: RSZ, layer: 1, pos: 2086
type: RSZ, layer: 1, pos: 2329
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 2100
type: RSZ, layer: 1, pos: 2446
type: RSZ, layer: 1, pos: 2507
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 395
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2089
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3186
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 2090
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2205
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 2071
type: RSZ, layer: 1, pos: 2118
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 2056
type: RSZ, layer: 1, pos: 2115
type: RSZ, layer: 1, pos: 3319
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 2106
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2976
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 2061
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2107
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2584
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 2181
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 3049
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2600
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 396
type: RSZ, layer: 1, pos: 125
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 333
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 332
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 2407
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 3063
type: RSZ, layer: 1, pos: 3425
type: RSZ, layer: 1, pos: 2595
type: RSZ, layer: 1, pos: 3097
type: RSZ, layer: 1, pos: 334
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 2044
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 3481
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2484
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 2258
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2254
type: RSZ, layer: 1, pos: 2259
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 350
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 145
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 149
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 192
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 337
type: RSZ, layer: 1, pos: 356
type: RSZ, layer: 1, pos: 357
type: RSZ, layer: 1, pos: 385
type: RSZ, layer: 1, pos: 404
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 2051
type: RSZ, layer: 1, pos: 2052
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 2068
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2084
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2108
type: RSZ, layer: 1, pos: 2109
type: RSZ, layer: 1, pos: 2110
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2127
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2144
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2209
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 2219
type: RSZ, layer: 1, pos: 2224
type: RSZ, layer: 1, pos: 2231
type: RSZ, layer: 1, pos: 2250
type: RSZ, layer: 1, pos: 2251
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2260
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2262
type: RSZ, layer: 1, pos: 2263
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 2452
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2622
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 2648
type: RSZ, layer: 1, pos: 2651
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 2692
type: RSZ, layer: 1, pos: 2695
type: RSZ, layer: 1, pos: 2940
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 3142
type: RSZ, layer: 1, pos: 3143
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 3326
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3503
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 3549
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 3591

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 379

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2539277, upper bound: 0.2530088
time: 28.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2539633, upper bound: 0.2529722
time: 42.73 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 74.42 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 74.42
Output dim: 4, lower bound: -0.2529597, upper bound: 0.2539768
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 74.42
Output dim: 4, lower bound: -0.2529960, upper bound: 0.2539373
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 74.42
Output dim: 4, lower bound: -0.2539277, upper bound: 0.2530088
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 74.42
Output dim: 4, lower bound: -0.2539633, upper bound: 0.2529722

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.3332100, -1.7664790, -4.3332100, -1.7664790, -0.9641521, 0.9641675
1: -3.5020158, -0.5655885, -3.5020158, -0.5655885, -1.2019460, 1.2017910
2: -1.8742058, -1.2430847, -1.8742058, -1.2430847, -0.2150533, 0.2149460
3: -0.3931611, 0.0556504, -0.3931611, 0.0556504, -0.3052554, 0.3053100
4: -0.9882243, 0.1546044, -0.9882243, 0.1546044, -0.9375443, 0.9376897
5: -0.6528150, -0.1970140, -0.6528150, -0.1970140, -0.2469686, 0.2469648
6: -2.7923801, -1.1410341, -2.7923801, -1.1410341, -0.5077614, 0.5080910
7: -0.7873561, 0.3647786, -0.7873561, 0.3647786, -1.0115379, 1.0115283
8: -4.1237435, -0.9955735, -4.1237435, -0.9955735, -1.2511561, 1.2509493
9: -4.9525442, -2.2417877, -4.9525442, -2.2417877, -1.0720069, 1.0718256

Time for backsubstitution: 2.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 275
type: RSZ, layer: 1, pos: 3002
type: RSZ, layer: 1, pos: 364
type: RSZ, layer: 1, pos: 2312
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2313
type: RSZ, layer: 1, pos: 2102
type: RSZ, layer: 1, pos: 2087
type: RSZ, layer: 1, pos: 3018
type: RSZ, layer: 1, pos: 3005
type: RSZ, layer: 1, pos: 2086
type: RSZ, layer: 1, pos: 2329
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 2100
type: RSZ, layer: 1, pos: 2446
type: RSZ, layer: 1, pos: 2507
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 395
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2089
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3186
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 2090
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2205
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 2071
type: RSZ, layer: 1, pos: 2118
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 2056
type: RSZ, layer: 1, pos: 2115
type: RSZ, layer: 1, pos: 3319
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 2106
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2976
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 2061
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2107
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2584
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 2181
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 3049
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2600
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 396
type: RSZ, layer: 1, pos: 125
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 333
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 332
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 2407
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 3063
type: RSZ, layer: 1, pos: 3425
type: RSZ, layer: 1, pos: 2595
type: RSZ, layer: 1, pos: 3097
type: RSZ, layer: 1, pos: 334
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 2044
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 3481
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2484
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 2258
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2254
type: RSZ, layer: 1, pos: 2259
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 350
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 145
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 149
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 192
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 337
type: RSZ, layer: 1, pos: 356
type: RSZ, layer: 1, pos: 357
type: RSZ, layer: 1, pos: 385
type: RSZ, layer: 1, pos: 404
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 2051
type: RSZ, layer: 1, pos: 2052
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 2068
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2084
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2108
type: RSZ, layer: 1, pos: 2109
type: RSZ, layer: 1, pos: 2110
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2127
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2144
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2209
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 2219
type: RSZ, layer: 1, pos: 2224
type: RSZ, layer: 1, pos: 2231
type: RSZ, layer: 1, pos: 2250
type: RSZ, layer: 1, pos: 2251
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2260
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2262
type: RSZ, layer: 1, pos: 2263
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 2452
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2622
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 2648
type: RSZ, layer: 1, pos: 2651
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 2692
type: RSZ, layer: 1, pos: 2695
type: RSZ, layer: 1, pos: 2940
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 3142
type: RSZ, layer: 1, pos: 3143
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 3326
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3503
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 3549
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 3591

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 275

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2528718, upper bound: 0.2539706
time: 344.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2529473, upper bound: 0.2538842
time: 157.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.3332100, -1.7664790, -4.3332100, -1.7664790, -0.9641432, 0.9641764
1: -3.5020158, -0.5655885, -3.5020158, -0.5655885, -1.2018573, 1.2018797
2: -1.8742058, -1.2430847, -1.8742058, -1.2430847, -0.2150569, 0.2149424
3: -0.3931611, 0.0556504, -0.3931611, 0.0556504, -0.3053106, 0.3052548
4: -0.9882243, 0.1546044, -0.9882243, 0.1546044, -0.9376564, 0.9375776
5: -0.6528150, -0.1970140, -0.6528150, -0.1970140, -0.2469746, 0.2469588
6: -2.7923801, -1.1410341, -2.7923801, -1.1410341, -0.5080770, 0.5077754
7: -0.7873561, 0.3647786, -0.7873561, 0.3647786, -1.0115383, 1.0115280
8: -4.1237435, -0.9955735, -4.1237435, -0.9955735, -1.2509202, 1.2511852
9: -4.9525442, -2.2417877, -4.9525442, -2.2417877, -1.0718560, 1.0719765

Time for backsubstitution: 2.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 275
type: RSZ, layer: 1, pos: 3002
type: RSZ, layer: 1, pos: 364
type: RSZ, layer: 1, pos: 2312
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2313
type: RSZ, layer: 1, pos: 2102
type: RSZ, layer: 1, pos: 2087
type: RSZ, layer: 1, pos: 3018
type: RSZ, layer: 1, pos: 3005
type: RSZ, layer: 1, pos: 2086
type: RSZ, layer: 1, pos: 2329
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 2100
type: RSZ, layer: 1, pos: 2446
type: RSZ, layer: 1, pos: 2507
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 395
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2089
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3186
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 2090
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2205
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 2071
type: RSZ, layer: 1, pos: 2118
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 2056
type: RSZ, layer: 1, pos: 2115
type: RSZ, layer: 1, pos: 3319
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 2106
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2976
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 2061
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2107
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2584
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 2181
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 3049
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2600
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 396
type: RSZ, layer: 1, pos: 125
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 333
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 332
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 2407
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 3063
type: RSZ, layer: 1, pos: 3425
type: RSZ, layer: 1, pos: 2595
type: RSZ, layer: 1, pos: 3097
type: RSZ, layer: 1, pos: 334
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 2044
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 3481
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2484
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 2258
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2254
type: RSZ, layer: 1, pos: 2259
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 350
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 145
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 149
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 192
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 337
type: RSZ, layer: 1, pos: 356
type: RSZ, layer: 1, pos: 357
type: RSZ, layer: 1, pos: 385
type: RSZ, layer: 1, pos: 404
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 2051
type: RSZ, layer: 1, pos: 2052
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 2068
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2084
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2108
type: RSZ, layer: 1, pos: 2109
type: RSZ, layer: 1, pos: 2110
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2127
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2144
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2209
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 2219
type: RSZ, layer: 1, pos: 2224
type: RSZ, layer: 1, pos: 2231
type: RSZ, layer: 1, pos: 2250
type: RSZ, layer: 1, pos: 2251
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2260
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2262
type: RSZ, layer: 1, pos: 2263
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 2452
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2622
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 2648
type: RSZ, layer: 1, pos: 2651
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 2692
type: RSZ, layer: 1, pos: 2695
type: RSZ, layer: 1, pos: 2940
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 3142
type: RSZ, layer: 1, pos: 3143
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 3326
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3503
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 3549
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 3591

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 275

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2529094, upper bound: 0.2539300
time: 252.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2529851, upper bound: 0.2538439
time: 318.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.3332100, -1.7664790, -4.3332100, -1.7664790, -0.9641764, 0.9641432
1: -3.5020158, -0.5655885, -3.5020158, -0.5655885, -1.2018797, 1.2018573
2: -1.8742058, -1.2430847, -1.8742058, -1.2430847, -0.2149424, 0.2150569
3: -0.3931611, 0.0556504, -0.3931611, 0.0556504, -0.3052548, 0.3053106
4: -0.9882243, 0.1546044, -0.9882243, 0.1546044, -0.9375776, 0.9376565
5: -0.6528150, -0.1970140, -0.6528150, -0.1970140, -0.2469588, 0.2469746
6: -2.7923801, -1.1410341, -2.7923801, -1.1410341, -0.5077754, 0.5080770
7: -0.7873561, 0.3647786, -0.7873561, 0.3647786, -1.0115279, 1.0115383
8: -4.1237435, -0.9955735, -4.1237435, -0.9955735, -1.2511851, 1.2509203
9: -4.9525442, -2.2417877, -4.9525442, -2.2417877, -1.0719765, 1.0718560

Time for backsubstitution: 3.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 275
type: RSZ, layer: 1, pos: 3002
type: RSZ, layer: 1, pos: 364
type: RSZ, layer: 1, pos: 2312
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2313
type: RSZ, layer: 1, pos: 2102
type: RSZ, layer: 1, pos: 2087
type: RSZ, layer: 1, pos: 3018
type: RSZ, layer: 1, pos: 3005
type: RSZ, layer: 1, pos: 2086
type: RSZ, layer: 1, pos: 2329
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 2100
type: RSZ, layer: 1, pos: 2446
type: RSZ, layer: 1, pos: 2507
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 395
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2089
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3186
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 2090
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2205
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 2071
type: RSZ, layer: 1, pos: 2118
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 2056
type: RSZ, layer: 1, pos: 2115
type: RSZ, layer: 1, pos: 3319
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 2106
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2976
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 2061
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2107
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2584
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 2181
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 3049
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2600
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 396
type: RSZ, layer: 1, pos: 125
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 333
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 332
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 2407
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 3063
type: RSZ, layer: 1, pos: 3425
type: RSZ, layer: 1, pos: 2595
type: RSZ, layer: 1, pos: 3097
type: RSZ, layer: 1, pos: 334
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 2044
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 3481
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2484
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 2258
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2254
type: RSZ, layer: 1, pos: 2259
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 350
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 145
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 149
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 192
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 337
type: RSZ, layer: 1, pos: 356
type: RSZ, layer: 1, pos: 357
type: RSZ, layer: 1, pos: 385
type: RSZ, layer: 1, pos: 404
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 2051
type: RSZ, layer: 1, pos: 2052
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 2068
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2084
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2108
type: RSZ, layer: 1, pos: 2109
type: RSZ, layer: 1, pos: 2110
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2127
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2144
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2209
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 2219
type: RSZ, layer: 1, pos: 2224
type: RSZ, layer: 1, pos: 2231
type: RSZ, layer: 1, pos: 2250
type: RSZ, layer: 1, pos: 2251
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2260
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2262
type: RSZ, layer: 1, pos: 2263
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 2452
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2622
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 2648
type: RSZ, layer: 1, pos: 2651
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 2692
type: RSZ, layer: 1, pos: 2695
type: RSZ, layer: 1, pos: 2940
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 3142
type: RSZ, layer: 1, pos: 3143
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 3326
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3503
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 3549
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 3591

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 275

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2538312, upper bound: 0.2529969
time: 261.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2539204, upper bound: 0.2529213
time: 80.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.3332100, -1.7664790, -4.3332100, -1.7664790, -0.9641675, 0.9641520
1: -3.5020158, -0.5655885, -3.5020158, -0.5655885, -1.2017908, 1.2019463
2: -1.8742058, -1.2430847, -1.8742058, -1.2430847, -0.2149460, 0.2150534
3: -0.3931611, 0.0556504, -0.3931611, 0.0556504, -0.3053100, 0.3052554
4: -0.9882243, 0.1546044, -0.9882243, 0.1546044, -0.9376898, 0.9375443
5: -0.6528150, -0.1970140, -0.6528150, -0.1970140, -0.2469648, 0.2469686
6: -2.7923801, -1.1410341, -2.7923801, -1.1410341, -0.5080910, 0.5077614
7: -0.7873561, 0.3647786, -0.7873561, 0.3647786, -1.0115283, 1.0115380
8: -4.1237435, -0.9955735, -4.1237435, -0.9955735, -1.2509491, 1.2511562
9: -4.9525442, -2.2417877, -4.9525442, -2.2417877, -1.0718255, 1.0720071

Time for backsubstitution: 3.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 275
type: RSZ, layer: 1, pos: 3002
type: RSZ, layer: 1, pos: 364
type: RSZ, layer: 1, pos: 2312
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2313
type: RSZ, layer: 1, pos: 2102
type: RSZ, layer: 1, pos: 2087
type: RSZ, layer: 1, pos: 3018
type: RSZ, layer: 1, pos: 3005
type: RSZ, layer: 1, pos: 2086
type: RSZ, layer: 1, pos: 2329
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 2100
type: RSZ, layer: 1, pos: 2446
type: RSZ, layer: 1, pos: 2507
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 395
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2089
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3186
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 2090
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2205
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 2071
type: RSZ, layer: 1, pos: 2118
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 2056
type: RSZ, layer: 1, pos: 2115
type: RSZ, layer: 1, pos: 3319
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 2106
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2976
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 2061
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2107
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2584
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 2181
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 3049
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2600
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 396
type: RSZ, layer: 1, pos: 125
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 333
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 332
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 2407
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 3063
type: RSZ, layer: 1, pos: 3425
type: RSZ, layer: 1, pos: 2595
type: RSZ, layer: 1, pos: 3097
type: RSZ, layer: 1, pos: 334
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 2044
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 3481
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2484
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 2258
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2254
type: RSZ, layer: 1, pos: 2259
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 350
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 145
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 149
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 192
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 337
type: RSZ, layer: 1, pos: 356
type: RSZ, layer: 1, pos: 357
type: RSZ, layer: 1, pos: 385
type: RSZ, layer: 1, pos: 404
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 2051
type: RSZ, layer: 1, pos: 2052
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 2068
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2084
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2108
type: RSZ, layer: 1, pos: 2109
type: RSZ, layer: 1, pos: 2110
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2127
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2144
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2209
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 2219
type: RSZ, layer: 1, pos: 2224
type: RSZ, layer: 1, pos: 2231
type: RSZ, layer: 1, pos: 2250
type: RSZ, layer: 1, pos: 2251
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2260
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2262
type: RSZ, layer: 1, pos: 2263
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 2452
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2622
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 2648
type: RSZ, layer: 1, pos: 2651
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 2692
type: RSZ, layer: 1, pos: 2695
type: RSZ, layer: 1, pos: 2940
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 3142
type: RSZ, layer: 1, pos: 3143
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 3326
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3503
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 3549
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 3591

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 275

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2538700, upper bound: 0.2529591
time: 172.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2539583, upper bound: 0.2528828
time: 307.15 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 482.49 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 482.49
Output dim: 4, lower bound: -0.2528718, upper bound: 0.2539706
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 482.49
Output dim: 4, lower bound: -0.2529473, upper bound: 0.2538842
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 482.49
Output dim: 4, lower bound: -0.2529094, upper bound: 0.2539300
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 482.49
Output dim: 4, lower bound: -0.2529851, upper bound: 0.2538439
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 482.49
Output dim: 4, lower bound: -0.2538312, upper bound: 0.2529969
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 482.49
Output dim: 4, lower bound: -0.2539204, upper bound: 0.2529213
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 482.49
Output dim: 4, lower bound: -0.2538700, upper bound: 0.2529591
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 482.49
Output dim: 4, lower bound: -0.2539583, upper bound: 0.2528828

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.3332100, -1.7664790, -4.3332100, -1.7664790, -0.9635915, 0.9636675
1: -3.5020158, -0.5655885, -3.5020158, -0.5655885, -1.2019031, 1.2017369
2: -1.8742058, -1.2430847, -1.8742058, -1.2430847, -0.2144757, 0.2142255
3: -0.3931611, 0.0556504, -0.3931611, 0.0556504, -0.3052420, 0.3052915
4: -0.9882243, 0.1546044, -0.9882243, 0.1546044, -0.9371495, 0.9374875
5: -0.6528150, -0.1970140, -0.6528150, -0.1970140, -0.2468262, 0.2467746
6: -2.7923801, -1.1410341, -2.7923801, -1.1410341, -0.5074734, 0.5078584
7: -0.7873561, 0.3647786, -0.7873561, 0.3647786, -1.0115449, 1.0115063
8: -4.1237435, -0.9955735, -4.1237435, -0.9955735, -1.2508568, 1.2507159
9: -4.9525442, -2.2417877, -4.9525442, -2.2417877, -1.0719812, 1.0717887

Time for backsubstitution: 3.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3002
type: RSZ, layer: 1, pos: 364
type: RSZ, layer: 1, pos: 2312
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2313
type: RSZ, layer: 1, pos: 2102
type: RSZ, layer: 1, pos: 2087
type: RSZ, layer: 1, pos: 3018
type: RSZ, layer: 1, pos: 3005
type: RSZ, layer: 1, pos: 2086
type: RSZ, layer: 1, pos: 2329
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 2100
type: RSZ, layer: 1, pos: 2446
type: RSZ, layer: 1, pos: 2507
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 395
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2089
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3186
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 2090
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2205
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 2071
type: RSZ, layer: 1, pos: 2118
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 2056
type: RSZ, layer: 1, pos: 2115
type: RSZ, layer: 1, pos: 3319
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 2106
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2976
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 2061
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2107
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2584
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 2181
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 3049
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2600
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 396
type: RSZ, layer: 1, pos: 125
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 333
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 332
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 2407
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 3063
type: RSZ, layer: 1, pos: 3425
type: RSZ, layer: 1, pos: 2595
type: RSZ, layer: 1, pos: 3097
type: RSZ, layer: 1, pos: 334
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 2044
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 3481
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2484
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 2258
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2254
type: RSZ, layer: 1, pos: 2259
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 350
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 145
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 149
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 192
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 337
type: RSZ, layer: 1, pos: 356
type: RSZ, layer: 1, pos: 357
type: RSZ, layer: 1, pos: 385
type: RSZ, layer: 1, pos: 404
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 2051
type: RSZ, layer: 1, pos: 2052
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 2068
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2084
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2108
type: RSZ, layer: 1, pos: 2109
type: RSZ, layer: 1, pos: 2110
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2127
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2144
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2209
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 2219
type: RSZ, layer: 1, pos: 2224
type: RSZ, layer: 1, pos: 2231
type: RSZ, layer: 1, pos: 2250
type: RSZ, layer: 1, pos: 2251
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2260
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2262
type: RSZ, layer: 1, pos: 2263
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 2452
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2622
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 2648
type: RSZ, layer: 1, pos: 2651
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 2692
type: RSZ, layer: 1, pos: 2695
type: RSZ, layer: 1, pos: 2940
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 3142
type: RSZ, layer: 1, pos: 3143
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 3326
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3503
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 3549
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 3591

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 3002

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2528567, upper bound: 0.2539395
time: 196.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2528438, upper bound: 0.2539536
time: 78.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.3332100, -1.7664790, -4.3332100, -1.7664790, -0.9636402, 0.9636070
1: -3.5020158, -0.5655885, -3.5020158, -0.5655885, -1.2018847, 1.2017479
2: -1.8742058, -1.2430847, -1.8742058, -1.2430847, -0.2143329, 0.2143677
3: -0.3931611, 0.0556504, -0.3931611, 0.0556504, -0.3052369, 0.3052952
4: -0.9882243, 0.1546044, -0.9882243, 0.1546044, -0.9373422, 0.9372948
5: -0.6528150, -0.1970140, -0.6528150, -0.1970140, -0.2467784, 0.2468085
6: -2.7923801, -1.1410341, -2.7923801, -1.1410341, -0.5075271, 0.5078031
7: -0.7873561, 0.3647786, -0.7873561, 0.3647786, -1.0115159, 1.0115271
8: -4.1237435, -0.9955735, -4.1237435, -0.9955735, -1.2509121, 1.2506498
9: -4.9525442, -2.2417877, -4.9525442, -2.2417877, -1.0719618, 1.0717999

Time for backsubstitution: 3.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3002
type: RSZ, layer: 1, pos: 364
type: RSZ, layer: 1, pos: 2312
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2313
type: RSZ, layer: 1, pos: 2102
type: RSZ, layer: 1, pos: 2087
type: RSZ, layer: 1, pos: 3018
type: RSZ, layer: 1, pos: 3005
type: RSZ, layer: 1, pos: 2086
type: RSZ, layer: 1, pos: 2329
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 2100
type: RSZ, layer: 1, pos: 2446
type: RSZ, layer: 1, pos: 2507
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 395
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2089
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3186
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 2090
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2205
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 2071
type: RSZ, layer: 1, pos: 2118
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 2056
type: RSZ, layer: 1, pos: 2115
type: RSZ, layer: 1, pos: 3319
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 2106
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2976
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 2061
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2107
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2584
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 2181
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 3049
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2600
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 396
type: RSZ, layer: 1, pos: 125
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 333
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 332
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 2407
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 3063
type: RSZ, layer: 1, pos: 3425
type: RSZ, layer: 1, pos: 2595
type: RSZ, layer: 1, pos: 3097
type: RSZ, layer: 1, pos: 334
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 2044
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 3481
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2484
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 2258
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2254
type: RSZ, layer: 1, pos: 2259
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 350
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 145
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 149
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 192
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 337
type: RSZ, layer: 1, pos: 356
type: RSZ, layer: 1, pos: 357
type: RSZ, layer: 1, pos: 385
type: RSZ, layer: 1, pos: 404
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 2051
type: RSZ, layer: 1, pos: 2052
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 2068
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2084
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2108
type: RSZ, layer: 1, pos: 2109
type: RSZ, layer: 1, pos: 2110
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2127
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2144
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2209
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 2219
type: RSZ, layer: 1, pos: 2224
type: RSZ, layer: 1, pos: 2231
type: RSZ, layer: 1, pos: 2250
type: RSZ, layer: 1, pos: 2251
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2260
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2262
type: RSZ, layer: 1, pos: 2263
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 2452
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2622
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 2648
type: RSZ, layer: 1, pos: 2651
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 2692
type: RSZ, layer: 1, pos: 2695
type: RSZ, layer: 1, pos: 2940
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 3142
type: RSZ, layer: 1, pos: 3143
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 3326
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3503
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 3549
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 3591

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 3002

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2529326, upper bound: 0.2538538
time: 447.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2529209, upper bound: 0.2538694
time: 81.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.3332100, -1.7664790, -4.3332100, -1.7664790, -0.9635826, 0.9636763
1: -3.5020158, -0.5655885, -3.5020158, -0.5655885, -1.2018144, 1.2018256
2: -1.8742058, -1.2430847, -1.8742058, -1.2430847, -0.2144793, 0.2142219
3: -0.3931611, 0.0556504, -0.3931611, 0.0556504, -0.3052972, 0.3052363
4: -0.9882243, 0.1546044, -0.9882243, 0.1546044, -0.9372616, 0.9373754
5: -0.6528150, -0.1970140, -0.6528150, -0.1970140, -0.2468322, 0.2467686
6: -2.7923801, -1.1410341, -2.7923801, -1.1410341, -0.5077891, 0.5075428
7: -0.7873561, 0.3647786, -0.7873561, 0.3647786, -1.0115454, 1.0115058
8: -4.1237435, -0.9955735, -4.1237435, -0.9955735, -1.2506208, 1.2509518
9: -4.9525442, -2.2417877, -4.9525442, -2.2417877, -1.0718303, 1.0719397

Time for backsubstitution: 6.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3002
type: RSZ, layer: 1, pos: 364
type: RSZ, layer: 1, pos: 2312
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2313
type: RSZ, layer: 1, pos: 2102
type: RSZ, layer: 1, pos: 2087
type: RSZ, layer: 1, pos: 3018
type: RSZ, layer: 1, pos: 3005
type: RSZ, layer: 1, pos: 2086
type: RSZ, layer: 1, pos: 2329
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 2100
type: RSZ, layer: 1, pos: 2446
type: RSZ, layer: 1, pos: 2507
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 395
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2089
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3186
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 2090
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2205
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 2071
type: RSZ, layer: 1, pos: 2118
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 2056
type: RSZ, layer: 1, pos: 2115
type: RSZ, layer: 1, pos: 3319
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 2106
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2976
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 2061
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2107
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2584
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 2181
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 3049
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2600
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 396
type: RSZ, layer: 1, pos: 125
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 333
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 332
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 2407
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 3063
type: RSZ, layer: 1, pos: 3425
type: RSZ, layer: 1, pos: 2595
type: RSZ, layer: 1, pos: 3097
type: RSZ, layer: 1, pos: 334
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 2044
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 3481
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2484
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 2258
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2254
type: RSZ, layer: 1, pos: 2259
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 350
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 145
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 149
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 192
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 337
type: RSZ, layer: 1, pos: 356
type: RSZ, layer: 1, pos: 357
type: RSZ, layer: 1, pos: 385
type: RSZ, layer: 1, pos: 404
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 2051
type: RSZ, layer: 1, pos: 2052
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 2068
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2084
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2108
type: RSZ, layer: 1, pos: 2109
type: RSZ, layer: 1, pos: 2110
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2127
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2144
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2209
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 2219
type: RSZ, layer: 1, pos: 2224
type: RSZ, layer: 1, pos: 2231
type: RSZ, layer: 1, pos: 2250
type: RSZ, layer: 1, pos: 2251
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2260
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2262
type: RSZ, layer: 1, pos: 2263
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 2452
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2622
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 2648
type: RSZ, layer: 1, pos: 2651
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 2692
type: RSZ, layer: 1, pos: 2695
type: RSZ, layer: 1, pos: 2940
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 3142
type: RSZ, layer: 1, pos: 3143
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 3326
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3503
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 3549
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 3591

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 1, pos: 3002

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2528961, upper bound: 0.2539046
time: 178.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2528833, upper bound: 0.2539141
time: 357.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.3332100, -1.7664790, -4.3332100, -1.7664790, -0.9636313, 0.9636158
1: -3.5020158, -0.5655885, -3.5020158, -0.5655885, -1.2017961, 1.2018368
2: -1.8742058, -1.2430847, -1.8742058, -1.2430847, -0.2143365, 0.2143642
3: -0.3931611, 0.0556504, -0.3931611, 0.0556504, -0.3052921, 0.3052400
4: -0.9882243, 0.1546044, -0.9882243, 0.1546044, -0.9374542, 0.9371827
5: -0.6528150, -0.1970140, -0.6528150, -0.1970140, -0.2467844, 0.2468025
6: -2.7923801, -1.1410341, -2.7923801, -1.1410341, -0.5078427, 0.5074875
7: -0.7873561, 0.3647786, -0.7873561, 0.3647786, -1.0115162, 1.0115266
8: -4.1237435, -0.9955735, -4.1237435, -0.9955735, -1.2506762, 1.2508857
9: -4.9525442, -2.2417877, -4.9525442, -2.2417877, -1.0718107, 1.0719508

Time for backsubstitution: 3.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3002
type: RSZ, layer: 1, pos: 364
type: RSZ, layer: 1, pos: 2312
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2313
type: RSZ, layer: 1, pos: 2102
type: RSZ, layer: 1, pos: 2087
type: RSZ, layer: 1, pos: 3018
type: RSZ, layer: 1, pos: 3005
type: RSZ, layer: 1, pos: 2086
type: RSZ, layer: 1, pos: 2329
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 2100
type: RSZ, layer: 1, pos: 2446
type: RSZ, layer: 1, pos: 2507
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 395
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2089
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3186
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 2090
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2205
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 2071
type: RSZ, layer: 1, pos: 2118
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 2056
type: RSZ, layer: 1, pos: 2115
type: RSZ, layer: 1, pos: 3319
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 2106
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2976
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 2061
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2107
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2584
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 2181
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 3049
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2600
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 396
type: RSZ, layer: 1, pos: 125
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 333
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 332
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 2407
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 3063
type: RSZ, layer: 1, pos: 3425
type: RSZ, layer: 1, pos: 2595
type: RSZ, layer: 1, pos: 3097
type: RSZ, layer: 1, pos: 334
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 2044
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 3481
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2484
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 2258
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2254
type: RSZ, layer: 1, pos: 2259
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 350
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 145
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 149
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 192
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 337
type: RSZ, layer: 1, pos: 356
type: RSZ, layer: 1, pos: 357
type: RSZ, layer: 1, pos: 385
type: RSZ, layer: 1, pos: 404
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 2051
type: RSZ, layer: 1, pos: 2052
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 2068
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2084
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2108
type: RSZ, layer: 1, pos: 2109
type: RSZ, layer: 1, pos: 2110
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2127
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2144
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2209
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 2219
type: RSZ, layer: 1, pos: 2224
type: RSZ, layer: 1, pos: 2231
type: RSZ, layer: 1, pos: 2250
type: RSZ, layer: 1, pos: 2251
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2260
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2262
type: RSZ, layer: 1, pos: 2263
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 2452
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2622
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 2648
type: RSZ, layer: 1, pos: 2651
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 2692
type: RSZ, layer: 1, pos: 2695
type: RSZ, layer: 1, pos: 2940
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 3142
type: RSZ, layer: 1, pos: 3143
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 3326
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3503
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 3549
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 3591

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 3002

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2529691, upper bound: 0.2538161
time: 359.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.2529581, upper bound: 0.2533827
time: 227.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.3332100, -1.7664790, -4.3332100, -1.7664790, -0.9636158, 0.9636313
1: -3.5020158, -0.5655885, -3.5020158, -0.5655885, -1.2018368, 1.2017961
2: -1.8742058, -1.2430847, -1.8742058, -1.2430847, -0.2143642, 0.2143364
3: -0.3931611, 0.0556504, -0.3931611, 0.0556504, -0.3052400, 0.3052921
4: -0.9882243, 0.1546044, -0.9882243, 0.1546044, -0.9371827, 0.9374542
5: -0.6528150, -0.1970140, -0.6528150, -0.1970140, -0.2468025, 0.2467844
6: -2.7923801, -1.1410341, -2.7923801, -1.1410341, -0.5074875, 0.5078427
7: -0.7873561, 0.3647786, -0.7873561, 0.3647786, -1.0115265, 1.0115163
8: -4.1237435, -0.9955735, -4.1237435, -0.9955735, -1.2508858, 1.2506760
9: -4.9525442, -2.2417877, -4.9525442, -2.2417877, -1.0719507, 1.0718108

Time for backsubstitution: 2.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3002
type: RSZ, layer: 1, pos: 364
type: RSZ, layer: 1, pos: 2312
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2313
type: RSZ, layer: 1, pos: 2102
type: RSZ, layer: 1, pos: 2087
type: RSZ, layer: 1, pos: 3018
type: RSZ, layer: 1, pos: 3005
type: RSZ, layer: 1, pos: 2086
type: RSZ, layer: 1, pos: 2329
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 2100
type: RSZ, layer: 1, pos: 2446
type: RSZ, layer: 1, pos: 2507
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 395
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2089
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3186
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 2090
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2205
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 2071
type: RSZ, layer: 1, pos: 2118
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 2056
type: RSZ, layer: 1, pos: 2115
type: RSZ, layer: 1, pos: 3319
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 2106
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2976
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 2061
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2107
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2584
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 2181
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 3049
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2600
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 396
type: RSZ, layer: 1, pos: 125
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 333
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 332
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 2407
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 3063
type: RSZ, layer: 1, pos: 3425
type: RSZ, layer: 1, pos: 2595
type: RSZ, layer: 1, pos: 3097
type: RSZ, layer: 1, pos: 334
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 2044
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 3481
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2484
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 2258
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2254
type: RSZ, layer: 1, pos: 2259
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 350
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 145
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 149
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 192
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 337
type: RSZ, layer: 1, pos: 356
type: RSZ, layer: 1, pos: 357
type: RSZ, layer: 1, pos: 385
type: RSZ, layer: 1, pos: 404
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 2051
type: RSZ, layer: 1, pos: 2052
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 2068
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2084
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2108
type: RSZ, layer: 1, pos: 2109
type: RSZ, layer: 1, pos: 2110
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2127
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2144
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2209
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 2219
type: RSZ, layer: 1, pos: 2224
type: RSZ, layer: 1, pos: 2231
type: RSZ, layer: 1, pos: 2250
type: RSZ, layer: 1, pos: 2251
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2260
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2262
type: RSZ, layer: 1, pos: 2263
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 2452
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2622
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 2648
type: RSZ, layer: 1, pos: 2651
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 2692
type: RSZ, layer: 1, pos: 2695
type: RSZ, layer: 1, pos: 2940
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 3142
type: RSZ, layer: 1, pos: 3143
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 3326
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3503
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 3549
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 3591

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 3002

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2538202, upper bound: 0.2529682
time: 372.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2538082, upper bound: 0.2529799
time: 171.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.3332100, -1.7664790, -4.3332100, -1.7664790, -0.9636763, 0.9635826
1: -3.5020158, -0.5655885, -3.5020158, -0.5655885, -1.2018256, 1.2018144
2: -1.8742058, -1.2430847, -1.8742058, -1.2430847, -0.2142219, 0.2144793
3: -0.3931611, 0.0556504, -0.3931611, 0.0556504, -0.3052363, 0.3052972
4: -0.9882243, 0.1546044, -0.9882243, 0.1546044, -0.9373753, 0.9372616
5: -0.6528150, -0.1970140, -0.6528150, -0.1970140, -0.2467686, 0.2468322
6: -2.7923801, -1.1410341, -2.7923801, -1.1410341, -0.5075428, 0.5077891
7: -0.7873561, 0.3647786, -0.7873561, 0.3647786, -1.0115058, 1.0115454
8: -4.1237435, -0.9955735, -4.1237435, -0.9955735, -1.2509519, 1.2506208
9: -4.9525442, -2.2417877, -4.9525442, -2.2417877, -1.0719396, 1.0718303

Time for backsubstitution: 2.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3002
type: RSZ, layer: 1, pos: 364
type: RSZ, layer: 1, pos: 2312
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2313
type: RSZ, layer: 1, pos: 2102
type: RSZ, layer: 1, pos: 2087
type: RSZ, layer: 1, pos: 3018
type: RSZ, layer: 1, pos: 3005
type: RSZ, layer: 1, pos: 2086
type: RSZ, layer: 1, pos: 2329
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 2100
type: RSZ, layer: 1, pos: 2446
type: RSZ, layer: 1, pos: 2507
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 395
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2089
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3186
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 2090
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2205
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 2071
type: RSZ, layer: 1, pos: 2118
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 2056
type: RSZ, layer: 1, pos: 2115
type: RSZ, layer: 1, pos: 3319
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 2106
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2976
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 2061
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2107
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2584
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 2181
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 3049
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2600
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 396
type: RSZ, layer: 1, pos: 125
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 333
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 332
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 2407
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 3063
type: RSZ, layer: 1, pos: 3425
type: RSZ, layer: 1, pos: 2595
type: RSZ, layer: 1, pos: 3097
type: RSZ, layer: 1, pos: 334
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 2044
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 3481
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2484
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 2258
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2254
type: RSZ, layer: 1, pos: 2259
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 350
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 145
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 149
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 192
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 337
type: RSZ, layer: 1, pos: 356
type: RSZ, layer: 1, pos: 357
type: RSZ, layer: 1, pos: 385
type: RSZ, layer: 1, pos: 404
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 2051
type: RSZ, layer: 1, pos: 2052
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 2068
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2084
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2108
type: RSZ, layer: 1, pos: 2109
type: RSZ, layer: 1, pos: 2110
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2127
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2144
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2209
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 2219
type: RSZ, layer: 1, pos: 2224
type: RSZ, layer: 1, pos: 2231
type: RSZ, layer: 1, pos: 2250
type: RSZ, layer: 1, pos: 2251
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2260
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2262
type: RSZ, layer: 1, pos: 2263
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 2452
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2622
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 2648
type: RSZ, layer: 1, pos: 2651
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 2692
type: RSZ, layer: 1, pos: 2695
type: RSZ, layer: 1, pos: 2940
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 3142
type: RSZ, layer: 1, pos: 3143
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 3326
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3503
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 3549
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 3591

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 3002

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2539039, upper bound: 0.2528968
time: 397.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2538929, upper bound: 0.2529071
time: 280.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.3332100, -1.7664790, -4.3332100, -1.7664790, -0.9636068, 0.9636402
1: -3.5020158, -0.5655885, -3.5020158, -0.5655885, -1.2017479, 1.2018847
2: -1.8742058, -1.2430847, -1.8742058, -1.2430847, -0.2143677, 0.2143329
3: -0.3931611, 0.0556504, -0.3931611, 0.0556504, -0.3052952, 0.3052368
4: -0.9882243, 0.1546044, -0.9882243, 0.1546044, -0.9372948, 0.9373421
5: -0.6528150, -0.1970140, -0.6528150, -0.1970140, -0.2468085, 0.2467784
6: -2.7923801, -1.1410341, -2.7923801, -1.1410341, -0.5078031, 0.5075271
7: -0.7873561, 0.3647786, -0.7873561, 0.3647786, -1.0115268, 1.0115159
8: -4.1237435, -0.9955735, -4.1237435, -0.9955735, -1.2506499, 1.2509120
9: -4.9525442, -2.2417877, -4.9525442, -2.2417877, -1.0717998, 1.0719618

Time for backsubstitution: 2.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3002
type: RSZ, layer: 1, pos: 364
type: RSZ, layer: 1, pos: 2312
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2313
type: RSZ, layer: 1, pos: 2102
type: RSZ, layer: 1, pos: 2087
type: RSZ, layer: 1, pos: 3018
type: RSZ, layer: 1, pos: 3005
type: RSZ, layer: 1, pos: 2086
type: RSZ, layer: 1, pos: 2329
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 2100
type: RSZ, layer: 1, pos: 2446
type: RSZ, layer: 1, pos: 2507
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 395
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2089
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3186
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 2090
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2205
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 2071
type: RSZ, layer: 1, pos: 2118
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 2056
type: RSZ, layer: 1, pos: 2115
type: RSZ, layer: 1, pos: 3319
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 2106
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2976
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 2061
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2107
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2584
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 2181
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 3049
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2600
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 396
type: RSZ, layer: 1, pos: 125
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 333
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 332
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 2407
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 3063
type: RSZ, layer: 1, pos: 3425
type: RSZ, layer: 1, pos: 2595
type: RSZ, layer: 1, pos: 3097
type: RSZ, layer: 1, pos: 334
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 2044
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 3481
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2484
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 2258
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2254
type: RSZ, layer: 1, pos: 2259
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 350
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 145
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 149
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 192
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 337
type: RSZ, layer: 1, pos: 356
type: RSZ, layer: 1, pos: 357
type: RSZ, layer: 1, pos: 385
type: RSZ, layer: 1, pos: 404
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 2051
type: RSZ, layer: 1, pos: 2052
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 2068
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2084
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2108
type: RSZ, layer: 1, pos: 2109
type: RSZ, layer: 1, pos: 2110
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2127
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2144
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2209
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 2219
type: RSZ, layer: 1, pos: 2224
type: RSZ, layer: 1, pos: 2231
type: RSZ, layer: 1, pos: 2250
type: RSZ, layer: 1, pos: 2251
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2260
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2262
type: RSZ, layer: 1, pos: 2263
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 2452
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2622
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 2648
type: RSZ, layer: 1, pos: 2651
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 2692
type: RSZ, layer: 1, pos: 2695
type: RSZ, layer: 1, pos: 2940
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 3142
type: RSZ, layer: 1, pos: 3143
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 3326
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3503
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 3549
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 3591

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 3002

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2538548, upper bound: 0.2529332
time: 31.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2538446, upper bound: 0.2529435
time: 188.16 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.3332100, -1.7664790, -4.3332100, -1.7664790, -0.9636674, 0.9635915
1: -3.5020158, -0.5655885, -3.5020158, -0.5655885, -1.2017369, 1.2019031
2: -1.8742058, -1.2430847, -1.8742058, -1.2430847, -0.2142255, 0.2144757
3: -0.3931611, 0.0556504, -0.3931611, 0.0556504, -0.3052915, 0.3052420
4: -0.9882243, 0.1546044, -0.9882243, 0.1546044, -0.9374874, 0.9371495
5: -0.6528150, -0.1970140, -0.6528150, -0.1970140, -0.2467746, 0.2468262
6: -2.7923801, -1.1410341, -2.7923801, -1.1410341, -0.5078584, 0.5074734
7: -0.7873561, 0.3647786, -0.7873561, 0.3647786, -1.0115062, 1.0115452
8: -4.1237435, -0.9955735, -4.1237435, -0.9955735, -1.2507160, 1.2508568
9: -4.9525442, -2.2417877, -4.9525442, -2.2417877, -1.0717887, 1.0719813

Time for backsubstitution: 2.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3002
type: RSZ, layer: 1, pos: 364
type: RSZ, layer: 1, pos: 2312
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 2313
type: RSZ, layer: 1, pos: 2102
type: RSZ, layer: 1, pos: 2087
type: RSZ, layer: 1, pos: 3018
type: RSZ, layer: 1, pos: 3005
type: RSZ, layer: 1, pos: 2086
type: RSZ, layer: 1, pos: 2329
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 2100
type: RSZ, layer: 1, pos: 2446
type: RSZ, layer: 1, pos: 2507
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 395
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2089
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3186
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 2090
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2205
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 2071
type: RSZ, layer: 1, pos: 2118
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 2056
type: RSZ, layer: 1, pos: 2115
type: RSZ, layer: 1, pos: 3319
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2970
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 2301
type: RSZ, layer: 1, pos: 2580
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 2106
type: RSZ, layer: 1, pos: 2673
type: RSZ, layer: 1, pos: 2976
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 2061
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2107
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 2280
type: RSZ, layer: 1, pos: 2584
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 2181
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 2287
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 3049
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 2600
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 396
type: RSZ, layer: 1, pos: 125
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 333
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 332
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 2407
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 3063
type: RSZ, layer: 1, pos: 3425
type: RSZ, layer: 1, pos: 2595
type: RSZ, layer: 1, pos: 3097
type: RSZ, layer: 1, pos: 334
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 2044
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 3481
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2163
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2484
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 2258
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2254
type: RSZ, layer: 1, pos: 2259
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 350
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 953
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 145
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 149
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 192
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 337
type: RSZ, layer: 1, pos: 356
type: RSZ, layer: 1, pos: 357
type: RSZ, layer: 1, pos: 385
type: RSZ, layer: 1, pos: 404
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 2051
type: RSZ, layer: 1, pos: 2052
type: RSZ, layer: 1, pos: 2053
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 2068
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2084
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2108
type: RSZ, layer: 1, pos: 2109
type: RSZ, layer: 1, pos: 2110
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2127
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2144
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2209
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 2219
type: RSZ, layer: 1, pos: 2224
type: RSZ, layer: 1, pos: 2231
type: RSZ, layer: 1, pos: 2250
type: RSZ, layer: 1, pos: 2251
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 2260
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2262
type: RSZ, layer: 1, pos: 2263
type: RSZ, layer: 1, pos: 2294
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2339
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2396
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 2452
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 2564
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2622
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 2648
type: RSZ, layer: 1, pos: 2651
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 2692
type: RSZ, layer: 1, pos: 2695
type: RSZ, layer: 1, pos: 2940
type: RSZ, layer: 1, pos: 2969
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3135
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 3142
type: RSZ, layer: 1, pos: 3143
type: RSZ, layer: 1, pos: 3144
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 3326
type: RSZ, layer: 1, pos: 3367
type: RSZ, layer: 1, pos: 3503
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 3549
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 3591

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 3002

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2539435, upper bound: 0.2528545
time: 502.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.2539311, upper bound: 0.2528699
time: 49.94 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 555.25 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 555.25
Output dim: 4, lower bound: -0.2528567, upper bound: 0.2539395
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 555.25
Output dim: 4, lower bound: -0.2528438, upper bound: 0.2539536
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 555.25
Output dim: 4, lower bound: -0.2529326, upper bound: 0.2538538
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 555.25
Output dim: 4, lower bound: -0.2529209, upper bound: 0.2538694
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 555.25
Output dim: 4, lower bound: -0.2528961, upper bound: 0.2539046
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 555.25
Output dim: 4, lower bound: -0.2528833, upper bound: 0.2539141
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 555.25
Output dim: 4, lower bound: -0.2529691, upper bound: 0.2538161
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 555.25
Output dim: 4, lower bound: -0.2529581, upper bound: 0.2533827
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 555.25
Output dim: 4, lower bound: -0.2538202, upper bound: 0.2529682
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 555.25
Output dim: 4, lower bound: -0.2538082, upper bound: 0.2529799
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 555.25
Output dim: 4, lower bound: -0.2539039, upper bound: 0.2528968
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 555.25
Output dim: 4, lower bound: -0.2538929, upper bound: 0.2529071
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 555.25
Output dim: 4, lower bound: -0.2538548, upper bound: 0.2529332
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 555.25
Output dim: 4, lower bound: -0.2538446, upper bound: 0.2529435
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 555.25
Output dim: 4, lower bound: -0.2539435, upper bound: 0.2528545
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 555.25
Output dim: 4, lower bound: -0.2539311, upper bound: 0.2528699

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 271.60 + 7293.32 = 7564.92 seconds
