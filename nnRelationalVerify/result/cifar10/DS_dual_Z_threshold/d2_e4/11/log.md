## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 11)
Time budget: 3600 seconds
Split limit: 100
Threshold: 0.1320998679


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-4.1708107, -1.5463735, -4.1708107, -1.5463735, -1.9036111, 1.9036112)
1: (-4.5101423, -0.4229836, -4.5101423, -0.4229836, -3.1152353, 3.1152356)
2: (-0.7876787, 0.0886384, -0.7876787, 0.0886384, -0.7659607, 0.7659606)
3: (-0.8389124, 0.0272182, -0.8389124, 0.0272182, -0.5533542, 0.5533543)
4: (-1.2802367, -0.1686895, -1.2802367, -0.1686895, -0.7999212, 0.7999212)
5: (-1.2698754, -0.3558336, -1.2698754, -0.3558336, -0.6029029, 0.6029029)
6: (-0.4843929, 0.4554565, -0.4843929, 0.4554565, -0.7734889, 0.7734888)
7: (-1.9408500, -0.9230059, -1.9408500, -0.9230059, -0.5432361, 0.5432360)
8: (-5.2866201, -2.2341690, -5.2866201, -2.2341690, -1.9068470, 1.9068470)
9: (-3.6426210, -0.8557649, -3.6426210, -0.8557649, -2.2256250, 2.2256250)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 9.56 + 109.05 = 118.60 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.1322178, upper bound: 0.1322309

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2651
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2186
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2541
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2109
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 342
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2782
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2993
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2994
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 3047
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 3062
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2976
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2978
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2320
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 3187
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2979
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2321
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 3466
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 261
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2806
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2959
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 2338
type: DSZ, layer: 1, pos: 2285
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 3328
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 3484
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 2718
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 2563
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 592
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 2963
type: DSZ, layer: 1, pos: 968
type: DSZ, layer: 1, pos: 585
type: DSZ, layer: 1, pos: 983
type: DSZ, layer: 1, pos: 984
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 648
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 55
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 471
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 659
type: DSZ, layer: 1, pos: 665
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2264
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2324
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2390
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2467
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2662
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3089
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 3344
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3373
type: DSZ, layer: 1, pos: 3530
type: DSZ, layer: 1, pos: 3531
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3586
type: DSZ, layer: 1, pos: 3587
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 3596

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 2424

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1318734, upper bound: 0.1320038
time: 89.76 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.1319963, upper bound: 0.1318796
time: 660.58 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 750.52 seconds
DS_DSZ1, status: Status.VERIFIED, split count: 1, time: 750.52
Output dim: 2, lower bound: -0.1318734, upper bound: 0.1320038
DS_DSZ2, status: Status.VERIFIED, split count: 1, time: 750.52
Output dim: 2, lower bound: -0.1319963, upper bound: 0.1318796

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 118.60 + 750.52 = 869.12 seconds
