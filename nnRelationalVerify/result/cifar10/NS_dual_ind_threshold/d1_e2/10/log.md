## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 10)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.0529606863


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-4.7559090, -4.1569338, -4.7559090, -4.1569338, -0.2038702, 0.2038702)
1: (-4.7411447, -4.2113256, -4.7411447, -4.2113256, -0.2005739, 0.2005739)
2: (-1.2580136, -1.0103903, -1.2580136, -1.0103903, -0.0831146, 0.0831146)
3: (-0.0328050, 0.3108093, -0.0328050, 0.3108093, -0.2911189, 0.2911189)
4: (-0.9833181, -0.6530223, -0.9833181, -0.6530223, -0.1150007, 0.1150007)
5: (-0.1361374, 0.3287530, -0.1361374, 0.3287530, -0.3120937, 0.3120938)
6: (0.5845805, 0.8929586, 0.5845805, 0.8929586, -0.1627387, 0.1627387)
7: (-1.1971323, -0.7447776, -1.1971323, -0.7447776, -0.0956259, 0.0956259)
8: (-5.1874294, -4.6035657, -5.1874294, -4.6035657, -0.2350359, 0.2350360)
9: (-5.3642797, -4.8095503, -5.3642797, -4.8095503, -0.2378700, 0.2378700)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.29 + 172.32 = 179.61 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0530126, upper bound: 0.0530138

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 322
type: A, layer: 1, pos: 385
type: A, layer: 1, pos: 420
type: A, layer: 1, pos: 3486
type: A, layer: 1, pos: 397
type: A, layer: 1, pos: 382
type: A, layer: 1, pos: 415
type: A, layer: 1, pos: 2599
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 2395
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 2587
type: A, layer: 1, pos: 2572
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 2134
type: A, layer: 1, pos: 3552
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 396
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 3556
type: A, layer: 1, pos: 363
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 2824
type: A, layer: 1, pos: 2825
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2411
type: A, layer: 1, pos: 2830
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 2798
type: A, layer: 1, pos: 2319
type: A, layer: 1, pos: 2995
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 2555
type: A, layer: 1, pos: 2797
type: A, layer: 1, pos: 3331
type: A, layer: 1, pos: 2810
type: A, layer: 1, pos: 3324
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 3338
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 2796
type: A, layer: 1, pos: 3354
type: A, layer: 1, pos: 2312
type: A, layer: 1, pos: 3562
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 2199
type: A, layer: 1, pos: 3339
type: A, layer: 1, pos: 3322
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 2306
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 2795
type: A, layer: 1, pos: 2132
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 3326
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 3020
type: A, layer: 1, pos: 3531
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 2892
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 432
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 2858
type: A, layer: 1, pos: 3559
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2877
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 3578
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 3534
type: A, layer: 1, pos: 2748
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 2213
type: A, layer: 1, pos: 2771
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2907
type: A, layer: 1, pos: 2211
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 2853
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 2897
type: A, layer: 1, pos: 2852
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 2081
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 2765
type: A, layer: 1, pos: 2293
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 2763
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2671
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 2029
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 2769
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 2028
type: A, layer: 1, pos: 2867
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 2749
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 2046
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 3117
type: A, layer: 1, pos: 2037
type: A, layer: 1, pos: 2215
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 164
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 443
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2084
type: A, layer: 1, pos: 2099
type: A, layer: 1, pos: 2129
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2534
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 3147
type: A, layer: 1, pos: 3366
type: A, layer: 1, pos: 3589
type: A, layer: 1, pos: 3590

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 322

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0530103, upper bound: 0.0526812
time: 202.10 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0530113, upper bound: 0.0530122
time: 454.93 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 657.10 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 657.10
Output dim: 5, lower bound: -0.0530103, upper bound: 0.0526812
NS_A2, status: Status.UNKNOWN, split count: 1, time: 657.10
Output dim: 5, lower bound: -0.0530113, upper bound: 0.0530122

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -4.7559066, -4.1574411, -4.7559071, -4.1573420, -0.2033529, 0.2032307
1: -4.7411437, -4.2119751, -4.7411437, -4.2118421, -0.2000177, 0.1998843
2: -1.2578644, -1.0103918, -1.2578938, -1.0103916, -0.0828679, 0.0829151
3: -0.0327064, 0.3104938, -0.0327251, 0.3105581, -0.2908132, 0.2907859
4: -0.9811209, -0.6530247, -0.9815707, -0.6530242, -0.1130053, 0.1133325
5: -0.1361093, 0.3274264, -0.1361144, 0.3276993, -0.3111337, 0.3109707
6: 0.5849970, 0.8929576, 0.5849254, 0.8929577, -0.1623360, 0.1623918
7: -1.1970551, -0.7469808, -1.1970696, -0.7465280, -0.0936878, 0.0935269
8: -5.1871324, -4.6035714, -5.1871934, -4.6035700, -0.2348551, 0.2348759
9: -5.3642230, -4.8100863, -5.3642340, -4.8099890, -0.2374472, 0.2373583

Time for backsubstitution: 5.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 385
type: B, layer: 1, pos: 420
type: B, layer: 1, pos: 3486
type: B, layer: 1, pos: 397
type: B, layer: 1, pos: 382
type: B, layer: 1, pos: 415
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 2395
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 2572
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 322
type: B, layer: 1, pos: 3552
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 396
type: B, layer: 1, pos: 2842
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 3556
type: B, layer: 1, pos: 363
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 2824
type: B, layer: 1, pos: 2825
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2411
type: B, layer: 1, pos: 2830
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 2798
type: B, layer: 1, pos: 2319
type: B, layer: 1, pos: 2995
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 2555
type: B, layer: 1, pos: 2797
type: B, layer: 1, pos: 3331
type: B, layer: 1, pos: 2810
type: B, layer: 1, pos: 3324
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 3338
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 2796
type: B, layer: 1, pos: 3354
type: B, layer: 1, pos: 2312
type: B, layer: 1, pos: 3562
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 2199
type: B, layer: 1, pos: 3339
type: B, layer: 1, pos: 3322
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 2306
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 2795
type: B, layer: 1, pos: 2132
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 3326
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 3020
type: B, layer: 1, pos: 3531
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 2892
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 432
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 2858
type: B, layer: 1, pos: 3559
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2877
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 3578
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 3534
type: B, layer: 1, pos: 2748
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2213
type: B, layer: 1, pos: 2771
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2907
type: B, layer: 1, pos: 2211
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 2853
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 2897
type: B, layer: 1, pos: 2852
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 2081
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 2765
type: B, layer: 1, pos: 2293
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 2763
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2671
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 2036
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2030
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 2051
type: B, layer: 1, pos: 2029
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2040
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 2769
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 2028
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 2749
type: B, layer: 1, pos: 2052
type: B, layer: 1, pos: 2046
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3117
type: B, layer: 1, pos: 2037
type: B, layer: 1, pos: 2215
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 164
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 443
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2084
type: B, layer: 1, pos: 2099
type: B, layer: 1, pos: 2129
type: B, layer: 1, pos: 2234
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2534
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 3147
type: B, layer: 1, pos: 3366
type: B, layer: 1, pos: 3589
type: B, layer: 1, pos: 3590

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 385

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0528094, upper bound: 0.0526815
time: 149.42 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0530105, upper bound: 0.0526848
time: 6.43 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -4.7565703, -4.1575556, -4.7559090, -4.1574702, -0.2044117, 0.2032985
1: -4.7420964, -4.2118492, -4.7411451, -4.2117887, -0.2012349, 0.1999675
2: -1.2579325, -1.0103704, -1.2578994, -1.0103908, -0.0830138, 0.0830656
3: -0.0339352, 0.3108077, -0.0327912, 0.3107995, -0.2921396, 0.2909135
4: -0.9832900, -0.6492574, -0.9832736, -0.6530234, -0.1134134, 0.1183411
5: -0.1386346, 0.3287336, -0.1361315, 0.3287368, -0.3142037, 0.3111703
6: 0.5837870, 0.8946518, 0.5848438, 0.8929582, -0.1646573, 0.1634177
7: -1.2010401, -0.7452278, -1.1971108, -0.7452185, -0.0989816, 0.0939844
8: -5.1873407, -4.6030903, -5.1873531, -4.6035686, -0.2349198, 0.2350465
9: -5.3661628, -4.8097277, -5.3642764, -4.8097048, -0.2394836, 0.2374428

Time for backsubstitution: 5.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 385
type: B, layer: 1, pos: 420
type: B, layer: 1, pos: 3486
type: B, layer: 1, pos: 397
type: B, layer: 1, pos: 382
type: B, layer: 1, pos: 415
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 2395
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 2572
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 322
type: B, layer: 1, pos: 3552
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 396
type: B, layer: 1, pos: 2842
type: B, layer: 1, pos: 2989
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 3556
type: B, layer: 1, pos: 363
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 2824
type: B, layer: 1, pos: 2825
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2411
type: B, layer: 1, pos: 2830
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 2798
type: B, layer: 1, pos: 2319
type: B, layer: 1, pos: 2995
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 2555
type: B, layer: 1, pos: 2797
type: B, layer: 1, pos: 3331
type: B, layer: 1, pos: 2810
type: B, layer: 1, pos: 3324
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 3338
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 2796
type: B, layer: 1, pos: 3354
type: B, layer: 1, pos: 2312
type: B, layer: 1, pos: 3562
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 2199
type: B, layer: 1, pos: 3339
type: B, layer: 1, pos: 3322
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 2306
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 2795
type: B, layer: 1, pos: 2132
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 3326
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 3020
type: B, layer: 1, pos: 3531
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2427
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 2892
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 432
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 2858
type: B, layer: 1, pos: 3559
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2877
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 3578
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 3534
type: B, layer: 1, pos: 2748
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 2213
type: B, layer: 1, pos: 2771
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2907
type: B, layer: 1, pos: 2211
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 2853
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 2897
type: B, layer: 1, pos: 2852
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 3345
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 2081
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 2765
type: B, layer: 1, pos: 2293
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 2763
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2671
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 2036
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2030
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 2051
type: B, layer: 1, pos: 2029
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2040
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 2769
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 2028
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 2749
type: B, layer: 1, pos: 2052
type: B, layer: 1, pos: 2046
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3117
type: B, layer: 1, pos: 2037
type: B, layer: 1, pos: 2215
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 164
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 443
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2084
type: B, layer: 1, pos: 2099
type: B, layer: 1, pos: 2129
type: B, layer: 1, pos: 2234
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2534
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 3147
type: B, layer: 1, pos: 3366
type: B, layer: 1, pos: 3589
type: B, layer: 1, pos: 3590

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 385

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0528102, upper bound: 0.0530117
time: 222.06 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0530103, upper bound: 0.0530134
time: 18.46 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 246.06 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 246.06
Output dim: 5, lower bound: -0.0528094, upper bound: 0.0526815
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 246.06
Output dim: 5, lower bound: -0.0530105, upper bound: 0.0526848
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 246.06
Output dim: 5, lower bound: -0.0528102, upper bound: 0.0530117
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 246.06
Output dim: 5, lower bound: -0.0530103, upper bound: 0.0530134

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -4.7552161, -4.1574421, -4.7550616, -4.1573429, -0.2031528, 0.2023203
1: -4.7402911, -4.2119765, -4.7400990, -4.2118425, -0.1995898, 0.1991589
2: -1.2578313, -1.0103918, -1.2578540, -1.0103916, -0.0826252, 0.0825803
3: -0.0326959, 0.3104820, -0.0327125, 0.3105436, -0.2900238, 0.2907642
4: -0.9810778, -0.6530248, -0.9815188, -0.6530244, -0.1129229, 0.1107762
5: -0.1361067, 0.3274153, -0.1361118, 0.3276857, -0.3102483, 0.3109571
6: 0.5849997, 0.8929575, 0.5849286, 0.8929578, -0.1623332, 0.1622660
7: -1.1969079, -0.7469808, -1.1969016, -0.7465281, -0.0935887, 0.0921693
8: -5.1871324, -4.6036506, -5.1871929, -4.6036673, -0.2323451, 0.2348394
9: -5.3632154, -4.8100877, -5.3629980, -4.8099890, -0.2368438, 0.2365540

Time for backsubstitution: 5.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 420
type: A, layer: 1, pos: 3486
type: A, layer: 1, pos: 397
type: A, layer: 1, pos: 382
type: A, layer: 1, pos: 415
type: A, layer: 1, pos: 2599
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 2395
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 2587
type: A, layer: 1, pos: 2572
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 2134
type: A, layer: 1, pos: 3552
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 396
type: A, layer: 1, pos: 385
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 3556
type: A, layer: 1, pos: 363
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 2824
type: A, layer: 1, pos: 2825
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2411
type: A, layer: 1, pos: 2830
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 2798
type: A, layer: 1, pos: 2319
type: A, layer: 1, pos: 2995
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 2555
type: A, layer: 1, pos: 2797
type: A, layer: 1, pos: 3331
type: A, layer: 1, pos: 2810
type: A, layer: 1, pos: 3324
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 3338
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 2796
type: A, layer: 1, pos: 3354
type: A, layer: 1, pos: 2312
type: A, layer: 1, pos: 3562
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 2199
type: A, layer: 1, pos: 3339
type: A, layer: 1, pos: 3322
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 2306
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 2795
type: A, layer: 1, pos: 2132
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 3326
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 3020
type: A, layer: 1, pos: 3531
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 2892
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 432
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 2858
type: A, layer: 1, pos: 3559
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2877
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 3578
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 3534
type: A, layer: 1, pos: 2748
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 2213
type: A, layer: 1, pos: 2771
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2907
type: A, layer: 1, pos: 2211
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 2853
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 2897
type: A, layer: 1, pos: 2852
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 2081
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 2765
type: A, layer: 1, pos: 2293
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 2763
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2671
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 2029
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 2769
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 2028
type: A, layer: 1, pos: 2867
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 2749
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 2046
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 3117
type: A, layer: 1, pos: 2037
type: A, layer: 1, pos: 2215
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 164
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 443
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2084
type: A, layer: 1, pos: 2099
type: A, layer: 1, pos: 2129
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2534
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 3147
type: A, layer: 1, pos: 3366
type: A, layer: 1, pos: 3589
type: A, layer: 1, pos: 3590

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 420

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0527551, upper bound: 0.0526839
time: 6.64 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0530097, upper bound: 0.0526806
time: 482.43 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -4.7564621, -4.1575727, -4.7557831, -4.1566758, -0.2035578, 0.2029264
1: -4.7418323, -4.2118559, -4.7408185, -4.2109404, -0.2006007, 0.1995226
2: -1.2576638, -1.0103709, -1.2574707, -1.0107884, -0.0824944, 0.0825556
3: -0.0338254, 0.3090125, -0.0302303, 0.3083948, -0.2896386, 0.2865407
4: -0.9822431, -0.6492708, -0.9818353, -0.6545012, -0.1107828, 0.1169375
5: -0.1385918, 0.3265843, -0.1336852, 0.3261062, -0.3115470, 0.3065773
6: 0.5843133, 0.8946469, 0.5855317, 0.8929081, -0.1640141, 0.1627108
7: -1.2005656, -0.7452366, -1.1959424, -0.7457675, -0.0975506, 0.0925145
8: -5.1873350, -4.6042051, -5.1861839, -4.6049371, -0.2334939, 0.2326039
9: -5.3657489, -4.8097353, -5.3636503, -4.8086705, -0.2387229, 0.2365657

Time for backsubstitution: 5.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 420
type: A, layer: 1, pos: 3486
type: A, layer: 1, pos: 397
type: A, layer: 1, pos: 382
type: A, layer: 1, pos: 415
type: A, layer: 1, pos: 2599
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 2395
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 2587
type: A, layer: 1, pos: 2572
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 2134
type: A, layer: 1, pos: 3552
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 396
type: A, layer: 1, pos: 385
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 2989
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 3556
type: A, layer: 1, pos: 363
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 2824
type: A, layer: 1, pos: 2825
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2411
type: A, layer: 1, pos: 2830
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 2798
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 2319
type: A, layer: 1, pos: 2995
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 2555
type: A, layer: 1, pos: 2797
type: A, layer: 1, pos: 3331
type: A, layer: 1, pos: 2810
type: A, layer: 1, pos: 3324
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 3338
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 2796
type: A, layer: 1, pos: 3354
type: A, layer: 1, pos: 2312
type: A, layer: 1, pos: 3562
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 2199
type: A, layer: 1, pos: 3339
type: A, layer: 1, pos: 3322
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 2306
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 2795
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 2132
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 3326
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 3020
type: A, layer: 1, pos: 3531
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2427
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 2892
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 432
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 2858
type: A, layer: 1, pos: 3559
type: A, layer: 1, pos: 2877
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 3578
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 3534
type: A, layer: 1, pos: 2748
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 2771
type: A, layer: 1, pos: 2213
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2907
type: A, layer: 1, pos: 2211
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 2853
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 2897
type: A, layer: 1, pos: 2852
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 3345
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 2081
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 2293
type: A, layer: 1, pos: 2765
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 2763
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2671
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 2029
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 2769
type: A, layer: 1, pos: 2867
type: A, layer: 1, pos: 2028
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 2749
type: A, layer: 1, pos: 2046
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 3117
type: A, layer: 1, pos: 2037
type: A, layer: 1, pos: 2215
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 164
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 443
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2084
type: A, layer: 1, pos: 2099
type: A, layer: 1, pos: 2129
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2534
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 3147
type: A, layer: 1, pos: 3366
type: A, layer: 1, pos: 3589
type: A, layer: 1, pos: 3590

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 420

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0525544, upper bound: 0.0530127
time: 7.72 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0528094, upper bound: 0.0530113
time: 281.07 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 179.61 + 1853.53 = 2033.14 seconds
