## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 1)
Time budget: 3600 seconds
Split limit: 100
Threshold: 0.30816332820000003


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-2.7559481, -0.5622158, -2.7559481, -0.5622158, -1.6495447, 1.6495446)
1: (-0.6287427, 0.7546692, -0.6287427, 0.7546692, -1.2209883, 1.2209884)
2: (-2.8935990, -1.3454305, -2.8935990, -1.3454305, -1.1057941, 1.1057940)
3: (-3.8761518, -1.3211582, -3.8761518, -1.3211582, -1.5255718, 1.5255718)
4: (-3.2793164, -1.4829080, -3.2793164, -1.4829080, -0.9594198, 0.9594197)
5: (-4.1635609, -1.4374789, -4.1635609, -1.4374789, -1.6863927, 1.6863928)
6: (-2.5404692, -0.2738461, -2.5404692, -0.2738461, -1.8316419, 1.8316418)
7: (-5.2319374, -2.2983932, -5.2319374, -2.2983932, -1.4391488, 1.4391488)
8: (-1.8323961, 1.0517855, -1.8323961, 1.0517855, -2.5217531, 2.5217533)
9: (-1.0878322, 0.7994137, -1.0878322, 0.7994137, -1.6605024, 1.6605023)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.75 + 57.42 = 65.17 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.3084699, upper bound: 0.3084745

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 349
type: DSZ, layer: 1, pos: 332
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 577
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 300
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2478
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 317
type: DSZ, layer: 1, pos: 277
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 301
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2614
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 540
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 3517
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 3242
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 3243
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3176
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 3202
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 350
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 266
type: DSZ, layer: 1, pos: 3232
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 451
type: DSZ, layer: 1, pos: 288
type: DSZ, layer: 1, pos: 3293
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 2683
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 657
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 3581
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 308
type: DSZ, layer: 1, pos: 3228
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 256
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 3177
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 3481
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 2845
type: DSZ, layer: 1, pos: 2843
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 1075
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 1076
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 1108
type: DSZ, layer: 1, pos: 1092
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 1093
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 1107
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 3215
type: DSZ, layer: 1, pos: 1074
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 1088
type: DSZ, layer: 1, pos: 1103
type: DSZ, layer: 1, pos: 1104
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 284
type: DSZ, layer: 1, pos: 310
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 314
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 357
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 647
type: DSZ, layer: 1, pos: 648
type: DSZ, layer: 1, pos: 652
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 856
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 1080
type: DSZ, layer: 1, pos: 1081
type: DSZ, layer: 1, pos: 1082
type: DSZ, layer: 1, pos: 1083
type: DSZ, layer: 1, pos: 1087
type: DSZ, layer: 1, pos: 1102
type: DSZ, layer: 1, pos: 1109
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 1118
type: DSZ, layer: 1, pos: 1119
type: DSZ, layer: 1, pos: 1120
type: DSZ, layer: 1, pos: 1121
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 1124
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2546
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 2563
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3220
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 3268
type: DSZ, layer: 1, pos: 3282
type: DSZ, layer: 1, pos: 3509
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 3576
type: DSZ, layer: 1, pos: 3585
type: DSZ, layer: 1, pos: 3598

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 349

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3081774, upper bound: 0.3081817
time: 241.80 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3081820, upper bound: 0.3081784
time: 418.24 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 660.12 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 660.12
Output dim: 2, lower bound: -0.3081774, upper bound: 0.3081817
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 660.12
Output dim: 2, lower bound: -0.3081820, upper bound: 0.3081784

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -2.7559481, -0.5622158, -2.7559481, -0.5622158, -1.6493983, 1.6493856
1: -0.6287427, 0.7546692, -0.6287427, 0.7546692, -1.2209586, 1.2209187
2: -2.8935990, -1.3454305, -2.8935990, -1.3454305, -1.1051863, 1.1052334
3: -3.8761518, -1.3211582, -3.8761518, -1.3211582, -1.5249293, 1.5252682
4: -3.2793164, -1.4829080, -3.2793164, -1.4829080, -0.9579616, 0.9580898
5: -4.1635609, -1.4374789, -4.1635609, -1.4374789, -1.6843958, 1.6848297
6: -2.5404692, -0.2738461, -2.5404692, -0.2738461, -1.8296583, 1.8298271
7: -5.2319374, -2.2983932, -5.2319374, -2.2983932, -1.4387414, 1.4388113
8: -1.8323961, 1.0517855, -1.8323961, 1.0517855, -2.5217445, 2.5217309
9: -1.0878322, 0.7994137, -1.0878322, 0.7994137, -1.6604902, 1.6605099

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 332
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 577
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 300
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2478
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 317
type: DSZ, layer: 1, pos: 277
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 301
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2614
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 540
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 3517
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 3242
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 3243
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3176
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 3202
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 350
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 266
type: DSZ, layer: 1, pos: 3232
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 451
type: DSZ, layer: 1, pos: 288
type: DSZ, layer: 1, pos: 3293
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 2683
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 657
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 3581
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 308
type: DSZ, layer: 1, pos: 3228
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 256
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 3177
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 3481
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 2845
type: DSZ, layer: 1, pos: 2843
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 1075
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 1076
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 1108
type: DSZ, layer: 1, pos: 1092
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 1093
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 1107
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 3215
type: DSZ, layer: 1, pos: 1074
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 1088
type: DSZ, layer: 1, pos: 1103
type: DSZ, layer: 1, pos: 1104
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 284
type: DSZ, layer: 1, pos: 310
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 314
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 357
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 647
type: DSZ, layer: 1, pos: 648
type: DSZ, layer: 1, pos: 652
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 856
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 1080
type: DSZ, layer: 1, pos: 1081
type: DSZ, layer: 1, pos: 1082
type: DSZ, layer: 1, pos: 1083
type: DSZ, layer: 1, pos: 1087
type: DSZ, layer: 1, pos: 1102
type: DSZ, layer: 1, pos: 1109
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 1118
type: DSZ, layer: 1, pos: 1119
type: DSZ, layer: 1, pos: 1120
type: DSZ, layer: 1, pos: 1121
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 1124
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2546
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 2563
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3220
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 3268
type: DSZ, layer: 1, pos: 3282
type: DSZ, layer: 1, pos: 3509
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 3576
type: DSZ, layer: 1, pos: 3585
type: DSZ, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 332

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3079017, upper bound: 0.3081556
time: 206.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3081486, upper bound: 0.3079126
time: 42.84 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -2.7559481, -0.5622158, -2.7559481, -0.5622158, -1.6493855, 1.6493984
1: -0.6287427, 0.7546692, -0.6287427, 0.7546692, -1.2209184, 1.2209587
2: -2.8935990, -1.3454305, -2.8935990, -1.3454305, -1.1052334, 1.1051863
3: -3.8761518, -1.3211582, -3.8761518, -1.3211582, -1.5252681, 1.5249294
4: -3.2793164, -1.4829080, -3.2793164, -1.4829080, -0.9580898, 0.9579617
5: -4.1635609, -1.4374789, -4.1635609, -1.4374789, -1.6848295, 1.6843958
6: -2.5404692, -0.2738461, -2.5404692, -0.2738461, -1.8298271, 1.8296586
7: -5.2319374, -2.2983932, -5.2319374, -2.2983932, -1.4388113, 1.4387414
8: -1.8323961, 1.0517855, -1.8323961, 1.0517855, -2.5217309, 2.5217447
9: -1.0878322, 0.7994137, -1.0878322, 0.7994137, -1.6605101, 1.6604902

Time for backsubstitution: 6.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 332
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 577
type: DSZ, layer: 1, pos: 3498
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 300
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 278
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2478
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 544
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 317
type: DSZ, layer: 1, pos: 277
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 301
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 3151
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2614
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 540
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 3517
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 3242
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 3243
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3176
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 3188
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 3202
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 350
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 266
type: DSZ, layer: 1, pos: 3232
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 658
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 451
type: DSZ, layer: 1, pos: 288
type: DSZ, layer: 1, pos: 3293
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 2683
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 657
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 2233
type: DSZ, layer: 1, pos: 2823
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 3581
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 308
type: DSZ, layer: 1, pos: 3228
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 256
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 3177
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 3233
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 3481
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 2845
type: DSZ, layer: 1, pos: 2843
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 1075
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 802
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3040
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 1076
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 1108
type: DSZ, layer: 1, pos: 1092
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 1093
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 1107
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 3215
type: DSZ, layer: 1, pos: 1074
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 1088
type: DSZ, layer: 1, pos: 1103
type: DSZ, layer: 1, pos: 1104
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 284
type: DSZ, layer: 1, pos: 310
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 314
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 357
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 647
type: DSZ, layer: 1, pos: 648
type: DSZ, layer: 1, pos: 652
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 856
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 871
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 1065
type: DSZ, layer: 1, pos: 1080
type: DSZ, layer: 1, pos: 1081
type: DSZ, layer: 1, pos: 1082
type: DSZ, layer: 1, pos: 1083
type: DSZ, layer: 1, pos: 1087
type: DSZ, layer: 1, pos: 1102
type: DSZ, layer: 1, pos: 1109
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 1118
type: DSZ, layer: 1, pos: 1119
type: DSZ, layer: 1, pos: 1120
type: DSZ, layer: 1, pos: 1121
type: DSZ, layer: 1, pos: 1122
type: DSZ, layer: 1, pos: 1124
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2546
type: DSZ, layer: 1, pos: 2549
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 2563
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3091
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3220
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 3268
type: DSZ, layer: 1, pos: 3282
type: DSZ, layer: 1, pos: 3509
type: DSZ, layer: 1, pos: 3524
type: DSZ, layer: 1, pos: 3576
type: DSZ, layer: 1, pos: 3585
type: DSZ, layer: 1, pos: 3598

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 332

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3079101, upper bound: 0.3081494
time: 55.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3081553, upper bound: 0.3079051
time: 209.48 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 270.79 seconds
DS_DSZ1_DSZ1, status: Status.VERIFIED, split count: 2, time: 270.79
Output dim: 2, lower bound: -0.3079017, upper bound: 0.3081556
DS_DSZ1_DSZ2, status: Status.VERIFIED, split count: 2, time: 270.79
Output dim: 2, lower bound: -0.3081486, upper bound: 0.3079126
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 270.79
Output dim: 2, lower bound: -0.3079101, upper bound: 0.3081494
DS_DSZ2_DSZ2, status: Status.VERIFIED, split count: 2, time: 270.79
Output dim: 2, lower bound: -0.3081553, upper bound: 0.3079051

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 65.17 + 1185.90 = 1251.07 seconds
