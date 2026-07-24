## Execution arguments:
Dataset: Dataset.CIFAR10
Network: onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 8)
Time budget: 7200 seconds
Split limit: 100
Threshold: 0.2956792248


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-5.2476301, -2.2661889, -5.2476301, -2.2661889, -2.3272653, 2.3272653)
1: (-5.5838933, -1.7482089, -5.5838933, -1.7482089, -3.2944527, 3.2944527)
2: (-1.4732211, -0.0145877, -1.4732211, -0.0145877, -1.2053856, 1.2053854)
3: (0.3578196, 1.0192342, 0.3578196, 1.0192342, -0.2846410, 0.2846410)
4: (-1.7616348, 0.1779025, -1.7616348, 0.1779025, -1.7208056, 1.7208058)
5: (0.0602400, 0.9661990, 0.0602400, 0.9661990, -0.6577548, 0.6577549)
6: (-1.7551060, -0.0277331, -1.7551060, -0.0277331, -1.4773127, 1.4773126)
7: (-1.1177582, 0.1418294, -1.1177582, 0.1418294, -0.9961020, 0.9961021)
8: (-5.6039734, -1.2400137, -5.6039734, -1.2400137, -3.5757875, 3.5757875)
9: (-5.0604444, -1.6978846, -5.0604444, -1.6978846, -2.6000023, 2.6000021)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 5.37 + 557.55 = 562.93 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.2973750, upper bound: 0.2973750

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2395
type: RSZ, layer: 1, pos: 3049
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2374
type: RSZ, layer: 1, pos: 2569
type: RSZ, layer: 1, pos: 2599
type: RSZ, layer: 1, pos: 2394
type: RSZ, layer: 1, pos: 3063
type: RSZ, layer: 1, pos: 2393
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 2584
type: RSZ, layer: 1, pos: 2574
type: RSZ, layer: 1, pos: 3486
type: RSZ, layer: 1, pos: 3502
type: RSZ, layer: 1, pos: 2348
type: RSZ, layer: 1, pos: 3023
type: RSZ, layer: 1, pos: 3053
type: RSZ, layer: 1, pos: 2604
type: RSZ, layer: 1, pos: 3048
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2344
type: RSZ, layer: 1, pos: 2590
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 3022
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3047
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2347
type: RSZ, layer: 1, pos: 2576
type: RSZ, layer: 1, pos: 2135
type: RSZ, layer: 1, pos: 2358
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 2345
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 2139
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2140
type: RSZ, layer: 1, pos: 3019
type: RSZ, layer: 1, pos: 2150
type: RSZ, layer: 1, pos: 2124
type: RSZ, layer: 1, pos: 2357
type: RSZ, layer: 1, pos: 3020
type: RSZ, layer: 1, pos: 2136
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 145
type: RSZ, layer: 1, pos: 2376
type: RSZ, layer: 1, pos: 98
type: RSZ, layer: 1, pos: 2154
type: RSZ, layer: 1, pos: 3067
type: RSZ, layer: 1, pos: 2601
type: RSZ, layer: 1, pos: 3052
type: RSZ, layer: 1, pos: 2141
type: RSZ, layer: 1, pos: 3084
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 2602
type: RSZ, layer: 1, pos: 2126
type: RSZ, layer: 1, pos: 115
type: RSZ, layer: 1, pos: 3032
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2556
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 3500
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2329
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 2617
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2106
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 2635
type: RSZ, layer: 1, pos: 2567
type: RSZ, layer: 1, pos: 2636
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 3078
type: RSZ, layer: 1, pos: 2540
type: RSZ, layer: 1, pos: 2554
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 116
type: RSZ, layer: 1, pos: 2108
type: RSZ, layer: 1, pos: 3081
type: RSZ, layer: 1, pos: 2539
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2343
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 3005
type: RSZ, layer: 1, pos: 2186
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 3077
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2342
type: RSZ, layer: 1, pos: 2093
type: RSZ, layer: 1, pos: 3009
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 101
type: RSZ, layer: 1, pos: 3073
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2651
type: RSZ, layer: 1, pos: 3018
type: RSZ, layer: 1, pos: 2650
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 2813
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 2402
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 324
type: RSZ, layer: 1, pos: 2180
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 2142
type: RSZ, layer: 1, pos: 2110
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 2798
type: RSZ, layer: 1, pos: 2799
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 2795
type: RSZ, layer: 1, pos: 2561
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 2546
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 2633
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 2796
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 3010
type: RSZ, layer: 1, pos: 2127
type: RSZ, layer: 1, pos: 2328
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 2327
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2797
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 2111
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 2643
type: RSZ, layer: 1, pos: 2178
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 2781
type: RSZ, layer: 1, pos: 2143
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 2632
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 2102
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2537
type: RSZ, layer: 1, pos: 2627
type: RSZ, layer: 1, pos: 2128
type: RSZ, layer: 1, pos: 2780
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 2552
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 2647
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 3276
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 2312
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 3533
type: RSZ, layer: 1, pos: 2512
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 2313
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 2112
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 2299
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 2061
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 2525
type: RSZ, layer: 1, pos: 2510
type: RSZ, layer: 1, pos: 2097
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 2075
type: RSZ, layer: 1, pos: 2074
type: RSZ, layer: 1, pos: 2509
type: RSZ, layer: 1, pos: 2524
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 3046
type: RSZ, layer: 1, pos: 2297
type: RSZ, layer: 1, pos: 2326
type: RSZ, layer: 1, pos: 2507
type: RSZ, layer: 1, pos: 2296
type: RSZ, layer: 1, pos: 2522
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 2098
type: RSZ, layer: 1, pos: 2295
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 2356
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 2536
type: RSZ, layer: 1, pos: 3277
type: RSZ, layer: 1, pos: 2521
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 2370
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2551
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 3031
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 2057
type: RSZ, layer: 1, pos: 2071
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 2086
type: RSZ, layer: 1, pos: 2566
type: RSZ, layer: 1, pos: 2581
type: RSZ, layer: 1, pos: 2101
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 2056
type: RSZ, layer: 1, pos: 2085
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2070
type: RSZ, layer: 1, pos: 2055
type: RSZ, layer: 1, pos: 2596
type: RSZ, layer: 1, pos: 2073
type: RSZ, layer: 1, pos: 3235
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 2131
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2100
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 2079
type: RSZ, layer: 1, pos: 2116
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 2058
type: RSZ, layer: 1, pos: 396
type: RSZ, layer: 1, pos: 2115
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 2611
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 2160
type: RSZ, layer: 1, pos: 2664
type: RSZ, layer: 1, pos: 2595
type: RSZ, layer: 1, pos: 2229
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 394
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 2610
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 2340
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 2492
type: RSZ, layer: 1, pos: 3321
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 2477
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 2268
type: RSZ, layer: 1, pos: 2266
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 2265
type: RSZ, layer: 1, pos: 2942
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 2476
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 2041
type: RSZ, layer: 1, pos: 2941
type: RSZ, layer: 1, pos: 2269
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2040
type: RSZ, layer: 1, pos: 2943
type: RSZ, layer: 1, pos: 2026
type: RSZ, layer: 1, pos: 2494
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 2233
type: RSZ, layer: 1, pos: 2270
type: RSZ, layer: 1, pos: 2230
type: RSZ, layer: 1, pos: 2025
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 368
type: RSZ, layer: 1, pos: 2215
type: RSZ, layer: 1, pos: 2232
type: RSZ, layer: 1, pos: 2231
type: RSZ, layer: 1, pos: 2495
type: RSZ, layer: 1, pos: 2480
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 2045
type: RSZ, layer: 1, pos: 2946
type: RSZ, layer: 1, pos: 2217
type: RSZ, layer: 1, pos: 2945
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 2030
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 2252
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2253
type: RSZ, layer: 1, pos: 310
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 3546
type: RSZ, layer: 1, pos: 2218
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 2250
type: RSZ, layer: 1, pos: 2251
type: RSZ, layer: 1, pos: 383
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 2226
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 2046
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 2031
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 2032
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2662
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 2995
type: RSZ, layer: 1, pos: 2034
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 2033
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 202
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 2499
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 2484
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 2934
type: RSZ, layer: 1, pos: 2928
type: RSZ, layer: 1, pos: 2926
type: RSZ, layer: 1, pos: 2927
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 252
type: RSZ, layer: 1, pos: 266
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 2035
type: RSZ, layer: 1, pos: 2036
type: RSZ, layer: 1, pos: 2037
type: RSZ, layer: 1, pos: 2038
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2081
type: RSZ, layer: 1, pos: 2082
type: RSZ, layer: 1, pos: 2083
type: RSZ, layer: 1, pos: 2084
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2114
type: RSZ, layer: 1, pos: 2129
type: RSZ, layer: 1, pos: 2144
type: RSZ, layer: 1, pos: 2159
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 2219
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 2221
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 2223
type: RSZ, layer: 1, pos: 2224
type: RSZ, layer: 1, pos: 2225
type: RSZ, layer: 1, pos: 2234
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2240
type: RSZ, layer: 1, pos: 2241
type: RSZ, layer: 1, pos: 2242
type: RSZ, layer: 1, pos: 2243
type: RSZ, layer: 1, pos: 2244
type: RSZ, layer: 1, pos: 2245
type: RSZ, layer: 1, pos: 2246
type: RSZ, layer: 1, pos: 2247
type: RSZ, layer: 1, pos: 2248
type: RSZ, layer: 1, pos: 2249
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2485
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2531
type: RSZ, layer: 1, pos: 2624
type: RSZ, layer: 1, pos: 2625
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 2639
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 2935
type: RSZ, layer: 1, pos: 2939
type: RSZ, layer: 1, pos: 3074
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 3386
type: RSZ, layer: 1, pos: 3401

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 2395

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2952322, upper bound: 0.2952345
time: 237.12 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2952322, upper bound: 0.2952341
time: 241.92 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 479.17 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 479.17
Output dim: 5, lower bound: -0.2952322, upper bound: 0.2952345
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 479.17
Output dim: 5, lower bound: -0.2952322, upper bound: 0.2952341

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 562.93 + 479.17 = 1042.10 seconds
