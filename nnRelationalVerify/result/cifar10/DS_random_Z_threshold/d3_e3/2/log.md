## Execution arguments:
Dataset: Dataset.CIFAR10
Network: onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 2)
Time budget: 7200 seconds
Split limit: 100
Threshold: 0.33368757275


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-5.1045642, -1.6512212, -5.1045642, -1.6512212, -2.8154287, 2.8154287)
1: (-5.7736254, -0.3074427, -5.7736254, -0.3074427, -4.5748572, 4.5748568)
2: (-1.4472741, -0.3155471, -1.4472741, -0.3155471, -0.6288937, 0.6288936)
3: (-0.6228578, 1.1057007, -0.6228578, 1.1057007, -1.5602981, 1.5602981)
4: (-2.4371905, -0.9968365, -2.4371905, -0.9968365, -0.8196620, 0.8196620)
5: (-0.7968712, 0.9236192, -0.7968712, 0.9236192, -1.4499497, 1.4499497)
6: (-1.7828910, 0.4266155, -1.7828910, 0.4266155, -1.5608702, 1.5608702)
7: (-1.4759959, 0.0730203, -1.4759959, 0.0730203, -1.0474219, 1.0474221)
8: (-6.3338752, -2.0932069, -6.3338752, -2.0932069, -2.9381361, 2.9381359)
9: (-3.1632032, 1.1139565, -3.1632032, 1.1139565, -3.7255883, 3.7255886)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 5.73 + 136.68 = 142.41 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.3338545, upper bound: 0.3338487

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 2199
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2619
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 2678
type: RSZ, layer: 1, pos: 2213
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 3127
type: RSZ, layer: 1, pos: 2450
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 2354
type: RSZ, layer: 1, pos: 318
type: RSZ, layer: 1, pos: 3461
type: RSZ, layer: 1, pos: 2559
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 320
type: RSZ, layer: 1, pos: 3382
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 2662
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 2437
type: RSZ, layer: 1, pos: 3424
type: RSZ, layer: 1, pos: 303
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 3520
type: RSZ, layer: 1, pos: 2419
type: RSZ, layer: 1, pos: 3081
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 376
type: RSZ, layer: 1, pos: 2581
type: RSZ, layer: 1, pos: 3514
type: RSZ, layer: 1, pos: 2083
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 2189
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 329
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 2102
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 3032
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 1102
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3094
type: RSZ, layer: 1, pos: 3140
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2517
type: RSZ, layer: 1, pos: 2393
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 472
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 2666
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 2563
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 345
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 2313
type: RSZ, layer: 1, pos: 2604
type: RSZ, layer: 1, pos: 2057
type: RSZ, layer: 1, pos: 2322
type: RSZ, layer: 1, pos: 2420
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 3245
type: RSZ, layer: 1, pos: 402
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 346
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1116
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 351
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 2448
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 2979
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2068
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2943
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 261
type: RSZ, layer: 1, pos: 2106
type: RSZ, layer: 1, pos: 2394
type: RSZ, layer: 1, pos: 2069
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 2611
type: RSZ, layer: 1, pos: 2620
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 2091
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2026
type: RSZ, layer: 1, pos: 3082
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 2569
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 3543
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 2304
type: RSZ, layer: 1, pos: 2162
type: RSZ, layer: 1, pos: 2603
type: RSZ, layer: 1, pos: 2670
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 2557
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2226
type: RSZ, layer: 1, pos: 304
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 1100
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 3246
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 2154
type: RSZ, layer: 1, pos: 2240
type: RSZ, layer: 1, pos: 2124
type: RSZ, layer: 1, pos: 116
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2290
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 3160
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 275
type: RSZ, layer: 1, pos: 3045
type: RSZ, layer: 1, pos: 2211
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 2334
type: RSZ, layer: 1, pos: 2090
type: RSZ, layer: 1, pos: 2033
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 2505
type: RSZ, layer: 1, pos: 2502
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 3238
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 2122
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 315
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 347
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 284
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 3225
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 2920
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 2123
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 2633
type: RSZ, layer: 1, pos: 3012
type: RSZ, layer: 1, pos: 2027
type: RSZ, layer: 1, pos: 2353
type: RSZ, layer: 1, pos: 348
type: RSZ, layer: 1, pos: 2465
type: RSZ, layer: 1, pos: 3083
type: RSZ, layer: 1, pos: 361
type: RSZ, layer: 1, pos: 3491
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 3485
type: RSZ, layer: 1, pos: 2520
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2735
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 3176
type: RSZ, layer: 1, pos: 2152
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 3194
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 2648
type: RSZ, layer: 1, pos: 2449
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 3228
type: RSZ, layer: 1, pos: 2487
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 2139
type: RSZ, layer: 1, pos: 3172
type: RSZ, layer: 1, pos: 3222
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 2964
type: RSZ, layer: 1, pos: 3125
type: RSZ, layer: 1, pos: 3067
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2206
type: RSZ, layer: 1, pos: 3484
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2692
type: RSZ, layer: 1, pos: 1064
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 3315
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 2995
type: RSZ, layer: 1, pos: 3113
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 3097
type: RSZ, layer: 1, pos: 2407
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 393
type: RSZ, layer: 1, pos: 2318
type: RSZ, layer: 1, pos: 2947
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 3124
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 3112
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 2944
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 2975
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 3173
type: RSZ, layer: 1, pos: 3445
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 3439
type: RSZ, layer: 1, pos: 2950
type: RSZ, layer: 1, pos: 2041
type: RSZ, layer: 1, pos: 2501
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 2451
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 2261
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 3272
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 2040
type: RSZ, layer: 1, pos: 2028
type: RSZ, layer: 1, pos: 3244
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 2241
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 2391
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2436
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 2066
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 2474
type: RSZ, layer: 1, pos: 2565
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2464
type: RSZ, layer: 1, pos: 2138
type: RSZ, layer: 1, pos: 2490
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 3031
type: RSZ, layer: 1, pos: 2082
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 3528
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 2463
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 172
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 3287
type: RSZ, layer: 1, pos: 2356
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 2406
type: RSZ, layer: 1, pos: 2618
type: RSZ, layer: 1, pos: 3536
type: RSZ, layer: 1, pos: 3158
type: RSZ, layer: 1, pos: 2532
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 2345
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 2615
type: RSZ, layer: 1, pos: 1086
type: RSZ, layer: 1, pos: 2137
type: RSZ, layer: 1, pos: 2116
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 2585
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 2553
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 3442
type: RSZ, layer: 1, pos: 2172
type: RSZ, layer: 1, pos: 2686
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 3089
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 2806
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 2423
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 2305
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 2921
type: RSZ, layer: 1, pos: 2208
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 3477
type: RSZ, layer: 1, pos: 3138
type: RSZ, layer: 1, pos: 2433
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 2743
type: RSZ, layer: 1, pos: 3056
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 3133
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 2489
type: RSZ, layer: 1, pos: 2188
type: RSZ, layer: 1, pos: 2452
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2954
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3271
type: RSZ, layer: 1, pos: 3028
type: RSZ, layer: 1, pos: 3139
type: RSZ, layer: 1, pos: 3499
type: RSZ, layer: 1, pos: 2392
type: RSZ, layer: 1, pos: 3043
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 224
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 3126
type: RSZ, layer: 1, pos: 3098
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1063
type: RSZ, layer: 1, pos: 3066
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 3110
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 490
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2408
type: RSZ, layer: 1, pos: 2363
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 2377
type: RSZ, layer: 1, pos: 248
type: RSZ, layer: 1, pos: 3057
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2434
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 3374
type: RSZ, layer: 1, pos: 2538
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 3273
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 2264
type: RSZ, layer: 1, pos: 2466
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2151
type: RSZ, layer: 1, pos: 2144
type: RSZ, layer: 1, pos: 333
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2435
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 2110
type: RSZ, layer: 1, pos: 2579
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 3230
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 3186
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 3111
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2121
type: RSZ, layer: 1, pos: 2454
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 2647
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 3096
type: RSZ, layer: 1, pos: 2438
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 2600
type: RSZ, layer: 1, pos: 164
type: RSZ, layer: 1, pos: 366
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 3141
type: RSZ, layer: 1, pos: 2399
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 2788
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 448
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 1115
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 2546
type: RSZ, layer: 1, pos: 2034
type: RSZ, layer: 1, pos: 3500
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 2696
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 2242
type: RSZ, layer: 1, pos: 2605

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2376

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3336826, upper bound: 0.3336760
time: 572.97 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3336826, upper bound: 0.3336749
time: 331.09 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 904.08 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 904.08
Output dim: 2, lower bound: -0.3336826, upper bound: 0.3336760
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 904.08
Output dim: 2, lower bound: -0.3336826, upper bound: 0.3336749

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 142.41 + 904.08 = 1046.49 seconds
