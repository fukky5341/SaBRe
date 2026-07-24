## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 11)
Time budget: 3600 seconds
Split limit: 100
Threshold: 0.0376491132


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3763310, 1.3763309)
1: (-4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7966783, 1.7966783)
2: (-0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4152753, 0.4152753)
3: (-0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2225757, 0.2225757)
4: (-0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7738838, 0.7738838)
5: (-0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2647811, 0.2647811)
6: (-0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1898307, 0.1898307)
7: (-0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0998714, 0.0998714)
8: (-6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9393454, 0.9393452)
9: (-4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7985216, 0.7985218)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 10.36 + 23.05 = 33.41 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0376758, upper bound: 0.0377053

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 3469
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 2359
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2360
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 2591
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 2126
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 2101
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 3050
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2120
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 2411
type: A, layer: 1, pos: 2106
type: A, layer: 1, pos: 2132
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 2590
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2345
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2179
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 3051
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2512
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 2127
type: A, layer: 1, pos: 2073
type: A, layer: 1, pos: 2620
type: A, layer: 1, pos: 2100
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 116
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2124
type: A, layer: 1, pos: 2081
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2046
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 2125
type: A, layer: 1, pos: 2518
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 2532
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2026
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2102
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 2517
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 2503
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 2082
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 2437
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 2488
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2475
type: A, layer: 1, pos: 2677
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 2160
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 2678
type: A, layer: 1, pos: 2083
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 2067
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 2985
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2955
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2045
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 2537
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 2672
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 2341
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 2322
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 2181
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2657
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 3112
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 2611
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2553
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 2647
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 2587
type: A, layer: 1, pos: 2226
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 2225
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2129
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2241
type: A, layer: 1, pos: 2242
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2399
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2609
type: A, layer: 1, pos: 2624
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2687
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2693
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 3138
type: A, layer: 1, pos: 3139
type: A, layer: 1, pos: 3144

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 335

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376711, upper bound: 0.0375091
time: 73.29 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376691, upper bound: 0.0376791
time: 86.21 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 159.57 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 159.57
Output dim: 2, lower bound: -0.0376711, upper bound: 0.0375091
NS_A2, status: Status.UNKNOWN, split count: 1, time: 159.57
Output dim: 2, lower bound: -0.0376691, upper bound: 0.0376791

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -4.7836905, -2.9990149, -4.7839441, -2.9989789, -1.3727632, 1.3757682
1: -4.8793564, -2.3038826, -4.8798242, -2.3037953, -1.7956648, 1.7959886
2: -0.5807434, -0.0945536, -0.5827504, -0.0941282, -0.4102430, 0.4121331
3: -0.5547605, -0.0470138, -0.5533537, -0.0469218, -0.2221246, 0.2215103
4: -0.6737903, 0.1198844, -0.6742312, 0.1213143, -0.7716258, 0.7691349
5: -0.8936034, -0.2065130, -0.8906016, -0.2064446, -0.2641647, 0.2630377
6: -0.2564807, 0.2045017, -0.2574881, 0.2045635, -0.1822189, 0.1823089
7: -0.8946810, -0.4139258, -0.8949679, -0.4137999, -0.0960375, 0.0968952
8: -6.1093612, -4.2808843, -6.1100812, -4.2807741, -0.9382395, 0.9385474
9: -4.2513685, -2.6379273, -4.2516117, -2.6378651, -0.7980121, 0.7981896

Time for backsubstitution: 7.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2605
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2360
type: B, layer: 1, pos: 2600
type: B, layer: 1, pos: 2591
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 2126
type: B, layer: 1, pos: 2087
type: B, layer: 1, pos: 2101
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 3050
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2120
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 335
type: B, layer: 1, pos: 2411
type: B, layer: 1, pos: 2106
type: B, layer: 1, pos: 2132
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2590
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 2539
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2179
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 3051
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 2127
type: B, layer: 1, pos: 2073
type: B, layer: 1, pos: 2620
type: B, layer: 1, pos: 2100
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 116
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2124
type: B, layer: 1, pos: 2081
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2046
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 2518
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 2532
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 2617
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2026
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2102
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 2517
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 145
type: B, layer: 1, pos: 2503
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 2082
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 2437
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 2488
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 2091
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2475
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 2160
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 2678
type: B, layer: 1, pos: 2083
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 2067
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 2203
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 2985
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 2666
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2955
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2045
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 2537
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 2672
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 2341
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 2322
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 2181
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2657
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 3112
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 2547
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2611
type: B, layer: 1, pos: 2231
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 2052
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2074
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2553
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 2647
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 2226
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 2225
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2129
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2234
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2241
type: B, layer: 1, pos: 2242
type: B, layer: 1, pos: 2243
type: B, layer: 1, pos: 2399
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2609
type: B, layer: 1, pos: 2624
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2687
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2693
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 3138
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3144

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3469

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376680, upper bound: 0.0374673
time: 8.66 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376698, upper bound: 0.0375296
time: 20.30 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -4.7840362, -2.9988515, -4.7840424, -2.9988503, -1.3749688, 1.3721416
1: -4.8798518, -2.3032284, -4.8798518, -2.3032246, -1.7966554, 1.7962153
2: -0.5827563, -0.0914280, -0.5827572, -0.0914240, -0.4152428, 0.4104466
3: -0.5533733, -0.0477498, -0.5533735, -0.0476012, -0.2220285, 0.2223207
4: -0.6760498, 0.1213156, -0.6760526, 0.1213156, -0.7732931, 0.7748028
5: -0.8906056, -0.2083977, -0.8906056, -0.2081168, -0.2637923, 0.2638171
6: -0.2616099, 0.2046187, -0.2616215, 0.2046694, -0.1892771, 0.1903192
7: -0.8957815, -0.4128183, -0.8957821, -0.4127600, -0.0995425, 0.0971095
8: -6.1101351, -4.2800245, -6.1101360, -4.2800236, -0.9389007, 0.9383919
9: -4.2516174, -2.6375980, -4.2516174, -2.6375978, -0.7985082, 0.7981336

Time for backsubstitution: 6.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2605
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2360
type: B, layer: 1, pos: 2600
type: B, layer: 1, pos: 2591
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 2126
type: B, layer: 1, pos: 2087
type: B, layer: 1, pos: 2101
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 3050
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2120
type: B, layer: 1, pos: 335
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 2411
type: B, layer: 1, pos: 2106
type: B, layer: 1, pos: 2132
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2590
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 2539
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2179
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 3051
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 2127
type: B, layer: 1, pos: 2073
type: B, layer: 1, pos: 2620
type: B, layer: 1, pos: 2100
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 116
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2124
type: B, layer: 1, pos: 2081
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2046
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 2518
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 2532
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 2617
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2026
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2102
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 2517
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 145
type: B, layer: 1, pos: 2503
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 2082
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 2437
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 2488
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 2091
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2475
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 2160
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 2678
type: B, layer: 1, pos: 2083
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 2067
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 2203
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 2985
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 2666
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2955
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2045
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 2537
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 2672
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 2341
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 2322
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 2181
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2657
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 3112
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 2547
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2611
type: B, layer: 1, pos: 2231
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 2052
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2074
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2553
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 2647
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 2226
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 2225
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2129
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2234
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2241
type: B, layer: 1, pos: 2242
type: B, layer: 1, pos: 2243
type: B, layer: 1, pos: 2399
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2609
type: B, layer: 1, pos: 2624
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2687
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2693
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 3138
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3144

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3469

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376685, upper bound: 0.0376267
time: 11.22 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376692, upper bound: 0.0376913
time: 9.19 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 26.59 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 26.59
Output dim: 2, lower bound: -0.0376680, upper bound: 0.0374673
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 26.59
Output dim: 2, lower bound: -0.0376698, upper bound: 0.0375296
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 26.59
Output dim: 2, lower bound: -0.0376685, upper bound: 0.0376267
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 26.59
Output dim: 2, lower bound: -0.0376692, upper bound: 0.0376913

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -4.7832899, -3.0000787, -4.7834826, -3.0001681, -1.3703415, 1.3735526
1: -4.8792410, -2.3038893, -4.8796926, -2.3038025, -1.7955451, 1.7958517
2: -0.5797566, -0.0946627, -0.5816224, -0.0942528, -0.4091164, 0.4108748
3: -0.5547548, -0.0473610, -0.5533473, -0.0473075, -0.2215747, 0.2209876
4: -0.6735176, 0.1186079, -0.6739419, 0.1198537, -0.7696537, 0.7673657
5: -0.8936032, -0.2071407, -0.8906015, -0.2071365, -0.2632824, 0.2622077
6: -0.2556548, 0.2045003, -0.2565442, 0.2045619, -0.1814766, 0.1814995
7: -0.8946686, -0.4140716, -0.8949546, -0.4139588, -0.0957427, 0.0966274
8: -6.1083746, -4.2809196, -6.1089835, -4.2808161, -0.9364949, 0.9367406
9: -4.2509966, -2.6379676, -4.2511883, -2.6379116, -0.7973136, 0.7974185

Time for backsubstitution: 6.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 2359
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2360
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 2591
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 2126
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 2101
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 3050
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2120
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 2411
type: A, layer: 1, pos: 2106
type: A, layer: 1, pos: 2132
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 2590
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2345
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2179
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 3051
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2512
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 2127
type: A, layer: 1, pos: 2073
type: A, layer: 1, pos: 2620
type: A, layer: 1, pos: 2100
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 116
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2124
type: A, layer: 1, pos: 2081
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2046
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 2125
type: A, layer: 1, pos: 2518
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 2532
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2026
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2102
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 2517
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 2503
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 3469
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 2082
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 2437
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 2488
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2475
type: A, layer: 1, pos: 2677
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2160
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 2678
type: A, layer: 1, pos: 2083
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 2067
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 2985
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2955
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2045
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 2537
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 2672
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 2341
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 2322
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 2181
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2657
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 3112
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 2611
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2553
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 2647
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 2587
type: A, layer: 1, pos: 2226
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 2225
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2129
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2241
type: A, layer: 1, pos: 2242
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2399
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2609
type: A, layer: 1, pos: 2624
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2687
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2693
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 3138
type: A, layer: 1, pos: 3139
type: A, layer: 1, pos: 3144

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0376138, upper bound: 0.0374598
time: 8.21 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376677, upper bound: 0.0374476
time: 247.18 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -4.7836595, -2.9992058, -4.7850180, -2.9991968, -1.3714256, 1.3774529
1: -4.8791499, -2.3038812, -4.8795886, -2.3038554, -1.7955317, 1.7958498
2: -0.5807378, -0.0945541, -0.5828406, -0.0928659, -0.4115003, 0.4120487
3: -0.5547575, -0.0470984, -0.5536422, -0.0468889, -0.2218787, 0.2217239
4: -0.6737852, 0.1198828, -0.6759052, 0.1213124, -0.7710732, 0.7711393
5: -0.8936034, -0.2065990, -0.8910278, -0.2065068, -0.2634989, 0.2633867
6: -0.2563436, 0.2045012, -0.2575861, 0.2056296, -0.1831932, 0.1819732
7: -0.8946795, -0.4140214, -0.8955115, -0.4138998, -0.0957684, 0.0976618
8: -6.1091237, -4.2808838, -6.1099195, -4.2799625, -0.9381205, 0.9377582
9: -4.2511091, -2.6379275, -4.2513161, -2.6376119, -0.7986115, 0.7974919

Time for backsubstitution: 6.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 2359
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2360
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 2591
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 2126
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 2101
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 3050
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2120
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 2411
type: A, layer: 1, pos: 2106
type: A, layer: 1, pos: 2132
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 2590
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2345
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2179
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 3051
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2512
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 2127
type: A, layer: 1, pos: 2073
type: A, layer: 1, pos: 2620
type: A, layer: 1, pos: 2100
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 116
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2124
type: A, layer: 1, pos: 2081
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2046
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 2125
type: A, layer: 1, pos: 2518
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 2532
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2026
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2102
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 2517
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 2503
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 3469
type: A, layer: 1, pos: 2082
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 2437
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 2488
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2475
type: A, layer: 1, pos: 2677
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 2160
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 2678
type: A, layer: 1, pos: 2083
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 2067
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 2985
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2955
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2045
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 2537
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 2672
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 2341
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 2322
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 2181
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2657
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 3112
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 2611
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2553
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 2647
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 2587
type: A, layer: 1, pos: 2226
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 2225
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2129
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2241
type: A, layer: 1, pos: 2242
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2399
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2609
type: A, layer: 1, pos: 2624
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2687
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2693
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 3138
type: A, layer: 1, pos: 3139
type: A, layer: 1, pos: 3144

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0376161, upper bound: 0.0375273
time: 117.23 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376696, upper bound: 0.0375321
time: 5.66 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -4.7836313, -2.9999158, -4.7835793, -3.0000393, -1.3725532, 1.3699181
1: -4.8797359, -2.3032346, -4.8797202, -2.3032322, -1.7965362, 1.7960796
2: -0.5817690, -0.0915359, -0.5816294, -0.0915477, -0.4141160, 0.4091892
3: -0.5533680, -0.0480967, -0.5533670, -0.0479869, -0.2214858, 0.2218022
4: -0.6757762, 0.1200395, -0.6757618, 0.1198554, -0.7713206, 0.7730325
5: -0.8906050, -0.2090241, -0.8906049, -0.2088066, -0.2629090, 0.2629953
6: -0.2607899, 0.2046176, -0.2606844, 0.2046679, -0.1885470, 0.1895318
7: -0.8957694, -0.4129636, -0.8957678, -0.4129184, -0.0992545, 0.0968416
8: -6.1091433, -4.2800608, -6.1090355, -4.2800646, -0.9371461, 0.9366050
9: -4.2512445, -2.6376388, -4.2511945, -2.6376433, -0.7978100, 0.7973628

Time for backsubstitution: 6.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 2359
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2360
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 2591
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 2126
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 2101
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 3050
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2120
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 2411
type: A, layer: 1, pos: 2106
type: A, layer: 1, pos: 2132
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 2590
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2345
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2179
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 3051
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2512
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 2127
type: A, layer: 1, pos: 2073
type: A, layer: 1, pos: 2620
type: A, layer: 1, pos: 2100
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 116
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2124
type: A, layer: 1, pos: 2081
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2046
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 2125
type: A, layer: 1, pos: 2518
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 2532
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2026
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2102
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 2517
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 2503
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 3469
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 2082
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 2437
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 2488
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2475
type: A, layer: 1, pos: 2677
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2160
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 2678
type: A, layer: 1, pos: 2083
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 2067
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 2985
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2955
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2045
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 2537
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 2672
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 2341
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 2322
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 2181
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2657
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 3112
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 2611
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2553
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 2647
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 2587
type: A, layer: 1, pos: 2226
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 2225
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2129
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2241
type: A, layer: 1, pos: 2242
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2399
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2609
type: A, layer: 1, pos: 2624
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2687
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2693
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 3138
type: A, layer: 1, pos: 3139
type: A, layer: 1, pos: 3144

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0376146, upper bound: 0.0376125
time: 66.48 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376684, upper bound: 0.0376074
time: 216.59 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -4.7840061, -2.9990425, -4.7851157, -2.9990680, -1.3736275, 1.3738563
1: -4.8796453, -2.3032284, -4.8796167, -2.3032842, -1.7965226, 1.7960756
2: -0.5827507, -0.0914287, -0.5828476, -0.0901608, -0.4165031, 0.4103621
3: -0.5533707, -0.0478342, -0.5536619, -0.0475690, -0.2217973, 0.2225477
4: -0.6760440, 0.1213139, -0.6777282, 0.1213140, -0.7727400, 0.7768073
5: -0.8906053, -0.2084866, -0.8910314, -0.2081821, -0.2631363, 0.2641801
6: -0.2614735, 0.2046182, -0.2617217, 0.2057374, -0.1902581, 0.1900035
7: -0.8957801, -0.4129139, -0.8963308, -0.4128600, -0.0992770, 0.0978922
8: -6.1098971, -4.2800245, -6.1099725, -4.2792125, -0.9388069, 0.9376336
9: -4.2513585, -2.6375988, -4.2513218, -2.6373427, -0.7991080, 0.7974359

Time for backsubstitution: 6.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 2359
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2360
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 2591
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 2126
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 2101
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 3050
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2120
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 2411
type: A, layer: 1, pos: 2106
type: A, layer: 1, pos: 2132
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 2590
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2345
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2179
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 3051
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2512
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 2127
type: A, layer: 1, pos: 2073
type: A, layer: 1, pos: 2620
type: A, layer: 1, pos: 2100
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 116
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2124
type: A, layer: 1, pos: 2081
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2046
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 2125
type: A, layer: 1, pos: 2518
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 2532
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2026
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2102
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 2517
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 2503
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 2082
type: A, layer: 1, pos: 3469
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 2437
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 2488
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2475
type: A, layer: 1, pos: 2677
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 2160
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 2678
type: A, layer: 1, pos: 2083
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 2067
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 2985
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2955
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2045
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 2537
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 2672
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 2341
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 2322
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 2181
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2657
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 3112
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 2611
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2553
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 2647
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 2587
type: A, layer: 1, pos: 2226
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 2225
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2129
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2241
type: A, layer: 1, pos: 2242
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2399
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2609
type: A, layer: 1, pos: 2624
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2687
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2693
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 3138
type: A, layer: 1, pos: 3139
type: A, layer: 1, pos: 3144

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0376163, upper bound: 0.0376183
time: 91.76 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376701, upper bound: 0.0376944
time: 8.47 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 106.84 seconds
NS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 106.84
Output dim: 2, lower bound: -0.0376138, upper bound: 0.0374598
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 106.84
Output dim: 2, lower bound: -0.0376677, upper bound: 0.0374476
NS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 106.84
Output dim: 2, lower bound: -0.0376161, upper bound: 0.0375273
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 106.84
Output dim: 2, lower bound: -0.0376696, upper bound: 0.0375321
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 106.84
Output dim: 2, lower bound: -0.0376146, upper bound: 0.0376125
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 106.84
Output dim: 2, lower bound: -0.0376684, upper bound: 0.0376074
NS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 106.84
Output dim: 2, lower bound: -0.0376163, upper bound: 0.0376183
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 106.84
Output dim: 2, lower bound: -0.0376701, upper bound: 0.0376944

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -4.7836800, -3.0000560, -4.7834826, -3.0001688, -1.3707188, 1.3733671
1: -4.8796124, -2.3038945, -4.8796926, -2.3038139, -1.7957389, 1.7954125
2: -0.5798050, -0.0936757, -0.5816196, -0.0942528, -0.4085811, 0.4118681
3: -0.5548621, -0.0471414, -0.5533466, -0.0473078, -0.2215294, 0.2212056
4: -0.6735492, 0.1189411, -0.6739382, 0.1198537, -0.7694970, 0.7676935
5: -0.8935981, -0.2065618, -0.8905941, -0.2071366, -0.2629465, 0.2627983
6: -0.2556685, 0.2051071, -0.2565420, 0.2045618, -0.1809853, 0.1821197
7: -0.8952752, -0.4140694, -0.8949546, -0.4139614, -0.0963565, 0.0961326
8: -6.1097775, -4.2808833, -6.1089835, -4.2808189, -0.9379228, 0.9358137
9: -4.2527990, -2.6379571, -4.2511883, -2.6379156, -0.7991322, 0.7960367

Time for backsubstitution: 6.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2605
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2360
type: B, layer: 1, pos: 2600
type: B, layer: 1, pos: 2591
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 2126
type: B, layer: 1, pos: 2087
type: B, layer: 1, pos: 2101
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 3050
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2120
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 335
type: B, layer: 1, pos: 2411
type: B, layer: 1, pos: 2106
type: B, layer: 1, pos: 2132
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2590
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 2539
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2179
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 3051
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 2127
type: B, layer: 1, pos: 2073
type: B, layer: 1, pos: 2620
type: B, layer: 1, pos: 2100
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 116
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2124
type: B, layer: 1, pos: 2081
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2046
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 2518
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 2532
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 2617
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2026
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2102
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 2517
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 145
type: B, layer: 1, pos: 2503
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 2082
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 2437
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 2488
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 2091
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2475
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2160
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 2678
type: B, layer: 1, pos: 2083
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 2067
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 2203
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 2985
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 2666
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2955
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2045
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 2537
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 2672
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 2341
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 2322
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 2181
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2657
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 3112
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 2547
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 2231
type: B, layer: 1, pos: 2611
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 2052
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2074
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2553
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 2647
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 2226
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 2225
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2129
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2234
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2241
type: B, layer: 1, pos: 2242
type: B, layer: 1, pos: 2243
type: B, layer: 1, pos: 2399
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2609
type: B, layer: 1, pos: 2624
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2687
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2693
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 3138
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3144

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2605

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0376223, upper bound: 0.0374555
time: 6.26 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376550, upper bound: 0.0374403
time: 163.46 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -4.7840486, -2.9991820, -4.7850170, -2.9991972, -1.3718026, 1.3772688
1: -4.8795209, -2.3038878, -4.8795886, -2.3038650, -1.7957249, 1.7954092
2: -0.5807866, -0.0935672, -0.5828381, -0.0928660, -0.4109656, 0.4130422
3: -0.5548651, -0.0468750, -0.5536411, -0.0468891, -0.2218332, 0.2219449
4: -0.6738267, 0.1202163, -0.6759012, 0.1213121, -0.7709246, 0.7714672
5: -0.8935987, -0.2060145, -0.8910207, -0.2065065, -0.2631630, 0.2639801
6: -0.2563572, 0.2051080, -0.2575842, 0.2056295, -0.1827023, 0.1825932
7: -0.8952858, -0.4140190, -0.8955115, -0.4139022, -0.0963823, 0.0971669
8: -6.1105375, -4.2808466, -6.1099195, -4.2799659, -0.9395546, 0.9368311
9: -4.2529116, -2.6379149, -4.2513151, -2.6376166, -0.8004301, 0.7961104

Time for backsubstitution: 6.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2605
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2360
type: B, layer: 1, pos: 2600
type: B, layer: 1, pos: 2591
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 2126
type: B, layer: 1, pos: 2087
type: B, layer: 1, pos: 2101
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 3050
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2120
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 335
type: B, layer: 1, pos: 2411
type: B, layer: 1, pos: 2106
type: B, layer: 1, pos: 2132
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2590
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 2539
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2179
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3051
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 2127
type: B, layer: 1, pos: 2073
type: B, layer: 1, pos: 2620
type: B, layer: 1, pos: 2100
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 116
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2124
type: B, layer: 1, pos: 2081
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2046
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 2518
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 2532
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 2617
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2026
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2102
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 2517
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 145
type: B, layer: 1, pos: 2503
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 2082
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 2437
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 2488
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 2091
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2475
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2160
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 2678
type: B, layer: 1, pos: 2083
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 2067
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 2203
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 2985
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 2666
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2955
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 2045
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 2537
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 2672
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 2341
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 2322
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 2181
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2657
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 3112
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 2547
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2231
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 2611
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 2052
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2074
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2553
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 2647
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 2226
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 2225
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2129
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2234
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2241
type: B, layer: 1, pos: 2242
type: B, layer: 1, pos: 2243
type: B, layer: 1, pos: 2399
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2609
type: B, layer: 1, pos: 2624
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2687
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2693
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 3138
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3144

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 2605

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0376253, upper bound: 0.0375235
time: 5.85 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376572, upper bound: 0.0375011
time: 138.34 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4.7840195, -2.9998920, -4.7835779, -3.0000410, -1.3729296, 1.3697348
1: -4.8801060, -2.3032408, -4.8797202, -2.3032422, -1.7967284, 1.7956398
2: -0.5818172, -0.0905489, -0.5816269, -0.0915477, -0.4135808, 0.4101826
3: -0.5534751, -0.0478767, -0.5533664, -0.0479870, -0.2214404, 0.2220202
4: -0.6758075, 0.1203717, -0.6757578, 0.1198553, -0.7711639, 0.7733611
5: -0.8906000, -0.2084457, -0.8905978, -0.2088065, -0.2625731, 0.2635860
6: -0.2608033, 0.2052244, -0.2606825, 0.2046679, -0.1880558, 0.1901519
7: -0.8963748, -0.4129612, -0.8957679, -0.4129209, -0.0998666, 0.0963467
8: -6.1105461, -4.2800245, -6.1090360, -4.2800679, -0.9385738, 0.9356779
9: -4.2530479, -2.6376278, -4.2511954, -2.6376486, -0.7996283, 0.7959803

Time for backsubstitution: 6.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2605
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2360
type: B, layer: 1, pos: 2600
type: B, layer: 1, pos: 2591
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 2126
type: B, layer: 1, pos: 2087
type: B, layer: 1, pos: 2101
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 3050
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2120
type: B, layer: 1, pos: 335
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 2411
type: B, layer: 1, pos: 2106
type: B, layer: 1, pos: 2132
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2590
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 2539
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2179
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 3051
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 2127
type: B, layer: 1, pos: 2073
type: B, layer: 1, pos: 2620
type: B, layer: 1, pos: 2100
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 116
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2124
type: B, layer: 1, pos: 2081
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2046
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 2518
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 2532
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 2617
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2026
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2102
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 2517
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 145
type: B, layer: 1, pos: 2503
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 2082
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 2437
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 2488
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 2091
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2475
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2160
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 2678
type: B, layer: 1, pos: 2083
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 2067
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 2203
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 2985
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 2666
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2955
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2045
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 2537
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 2672
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 2341
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 2322
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 2181
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2657
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 3112
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 2547
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 2231
type: B, layer: 1, pos: 2611
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 2052
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2074
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2553
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 2647
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 2226
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 2225
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2129
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2234
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2241
type: B, layer: 1, pos: 2242
type: B, layer: 1, pos: 2243
type: B, layer: 1, pos: 2399
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2609
type: B, layer: 1, pos: 2624
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2687
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2693
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 3138
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3144

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2605

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0376225, upper bound: 0.0375927
time: 158.60 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376555, upper bound: 0.0376153
time: 6.27 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.7843943, -2.9990189, -4.7851143, -2.9990683, -1.3740044, 1.3736718
1: -4.8800149, -2.3032341, -4.8796167, -2.3032947, -1.7967155, 1.7956357
2: -0.5827992, -0.0904416, -0.5828450, -0.0901612, -0.4159682, 0.4113555
3: -0.5534778, -0.0476111, -0.5536610, -0.0475694, -0.2217523, 0.2227687
4: -0.6760858, 0.1216468, -0.6777239, 0.1213139, -0.7725919, 0.7771366
5: -0.8906006, -0.2079024, -0.8910244, -0.2081822, -0.2628006, 0.2647736
6: -0.2614875, 0.2052251, -0.2617196, 0.2057374, -0.1897672, 0.1906235
7: -0.8963863, -0.4129116, -0.8963308, -0.4128625, -0.0998892, 0.0973973
8: -6.1113105, -4.2799888, -6.1099730, -4.2792153, -0.9402412, 0.9367058
9: -4.2531610, -2.6375868, -4.2513218, -2.6373477, -0.8009263, 0.7960546

Time for backsubstitution: 6.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2605
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2360
type: B, layer: 1, pos: 2600
type: B, layer: 1, pos: 2591
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 2126
type: B, layer: 1, pos: 2087
type: B, layer: 1, pos: 2101
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 3050
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2120
type: B, layer: 1, pos: 335
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 2411
type: B, layer: 1, pos: 2106
type: B, layer: 1, pos: 2132
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2590
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 2539
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2179
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3051
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 2127
type: B, layer: 1, pos: 2073
type: B, layer: 1, pos: 2620
type: B, layer: 1, pos: 2100
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 116
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2124
type: B, layer: 1, pos: 2081
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2046
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 2518
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 2532
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 2617
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2026
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2102
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 2517
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 145
type: B, layer: 1, pos: 2503
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 2082
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 2437
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 2488
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 2091
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2475
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2160
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 2678
type: B, layer: 1, pos: 2083
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 2067
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 2203
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 2985
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 2666
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2955
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 2045
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 2537
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 2672
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 2341
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 2322
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 2181
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2657
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 3112
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 2547
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2231
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 2611
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 2052
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2074
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2553
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 2647
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 2226
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 2225
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2129
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2234
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2241
type: B, layer: 1, pos: 2242
type: B, layer: 1, pos: 2243
type: B, layer: 1, pos: 2399
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2609
type: B, layer: 1, pos: 2624
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2687
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2693
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 3138
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3144

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2605

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376251, upper bound: 0.0376649
time: 128.12 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376566, upper bound: 0.0376741
time: 9.47 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 144.37 seconds
NS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 144.37
Output dim: 2, lower bound: -0.0376223, upper bound: 0.0374555
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 144.37
Output dim: 2, lower bound: -0.0376550, upper bound: 0.0374403
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 144.37
Output dim: 2, lower bound: -0.0376253, upper bound: 0.0375235
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 144.37
Output dim: 2, lower bound: -0.0376572, upper bound: 0.0375011
NS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 144.37
Output dim: 2, lower bound: -0.0376225, upper bound: 0.0375927
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 144.37
Output dim: 2, lower bound: -0.0376555, upper bound: 0.0376153
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 144.37
Output dim: 2, lower bound: -0.0376251, upper bound: 0.0376649
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 144.37
Output dim: 2, lower bound: -0.0376566, upper bound: 0.0376741

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -4.7785740, -3.0000575, -4.7778101, -2.9906416, -1.3799603, 1.3669035
1: -4.8750238, -2.3038955, -4.8749981, -2.2889500, -1.8142135, 1.7869821
2: -0.5797994, -0.0939559, -0.5822929, -0.0945764, -0.4082245, 0.4124877
3: -0.5548552, -0.0478016, -0.5554266, -0.0479913, -0.2201318, 0.2238616
4: -0.6735404, 0.1188251, -0.6747955, 0.1197224, -0.7691830, 0.7685292
5: -0.8935977, -0.2074081, -0.8935186, -0.2080411, -0.2608737, 0.2670825
6: -0.2556441, 0.2049849, -0.2567365, 0.2044216, -0.1808479, 0.1826402
7: -0.8952751, -0.4145446, -0.8966789, -0.4144644, -0.0952250, 0.0986208
8: -6.1045046, -4.2808895, -6.1029038, -4.2707186, -0.9559227, 0.9272373
9: -4.2498446, -2.6379578, -4.2477794, -2.6291144, -0.8134853, 0.7892901

Time for backsubstitution: 6.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2359
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2360
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 2591
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 2126
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 2101
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 3050
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2120
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 2411
type: A, layer: 1, pos: 2106
type: A, layer: 1, pos: 2132
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 2590
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2345
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2179
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 3051
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2512
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 2127
type: A, layer: 1, pos: 2073
type: A, layer: 1, pos: 2620
type: A, layer: 1, pos: 2100
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 116
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2124
type: A, layer: 1, pos: 2081
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2046
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 2125
type: A, layer: 1, pos: 2518
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 2532
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2026
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2102
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 2517
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 2503
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 3469
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 2082
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 2437
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 2488
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2475
type: A, layer: 1, pos: 2677
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2160
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 2678
type: A, layer: 1, pos: 2083
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 2067
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 2985
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2955
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2045
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 2537
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 2672
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 2341
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 2322
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 2181
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2657
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 3112
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 2611
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2553
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 2647
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 2587
type: A, layer: 1, pos: 2226
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 2225
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2129
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2241
type: A, layer: 1, pos: 2242
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2399
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2609
type: A, layer: 1, pos: 2624
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2687
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2693
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 3138
type: A, layer: 1, pos: 3139
type: A, layer: 1, pos: 3144

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2359

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0376124, upper bound: 0.0374323
time: 8.29 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0376361, upper bound: 0.0374353
time: 7.85 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4.7789435, -2.9991837, -4.7793446, -2.9896698, -1.3810470, 1.3708034
1: -4.8749309, -2.3038893, -4.8748956, -2.2890024, -1.8142025, 1.7869775
2: -0.5807804, -0.0938473, -0.5835103, -0.0931893, -0.4106083, 0.4136608
3: -0.5548581, -0.0475391, -0.5557215, -0.0475763, -0.2204327, 0.2245976
4: -0.6738074, 0.1201006, -0.6767462, 0.1211811, -0.7706016, 0.7722927
5: -0.8935987, -0.2068670, -0.8939449, -0.2074177, -0.2610869, 0.2682613
6: -0.2563328, 0.2049860, -0.2577775, 0.2054893, -0.1825644, 0.1831133
7: -0.8952855, -0.4144943, -0.8972358, -0.4144053, -0.0952506, 0.0996553
8: -6.1052537, -4.2808537, -6.1038275, -4.2698655, -0.9575480, 0.9282492
9: -4.2499585, -2.6379163, -4.2479072, -2.6288157, -0.8147824, 0.7893641

Time for backsubstitution: 7.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2359
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2360
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 2591
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 2126
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 2101
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 3050
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2120
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 2411
type: A, layer: 1, pos: 2106
type: A, layer: 1, pos: 2132
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 2590
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2345
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2179
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 3051
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2512
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 2127
type: A, layer: 1, pos: 2073
type: A, layer: 1, pos: 2620
type: A, layer: 1, pos: 2100
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 116
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2124
type: A, layer: 1, pos: 2081
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2046
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 2125
type: A, layer: 1, pos: 2518
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 2532
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2026
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2102
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 2517
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 2503
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 3469
type: A, layer: 1, pos: 2082
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 2437
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 2488
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2475
type: A, layer: 1, pos: 2677
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2160
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 2678
type: A, layer: 1, pos: 2083
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 2067
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 2985
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2955
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2045
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 2537
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 2672
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 2341
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 2322
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 2181
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2657
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 3112
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 2611
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2553
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 2647
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 2587
type: A, layer: 1, pos: 2226
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 2225
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2129
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2241
type: A, layer: 1, pos: 2242
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2399
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2609
type: A, layer: 1, pos: 2624
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2687
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2693
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 3138
type: A, layer: 1, pos: 3139
type: A, layer: 1, pos: 3144

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2359

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0376137, upper bound: 0.0374855
time: 124.85 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0376390, upper bound: 0.0374202
time: 167.26 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -4.7789145, -2.9998932, -4.7779055, -2.9905150, -1.3821700, 1.3632703
1: -4.8755174, -2.3032413, -4.8750267, -2.2883787, -1.8152044, 1.7872090
2: -0.5818116, -0.0908291, -0.5822999, -0.0918713, -0.4132242, 0.4108020
3: -0.5534676, -0.0485371, -0.5554460, -0.0486703, -0.2200446, 0.2246762
4: -0.6757988, 0.1202563, -0.6766163, 0.1197243, -0.7708497, 0.7741960
5: -0.8906002, -0.2092918, -0.8935220, -0.2097106, -0.2605002, 0.2678701
6: -0.2607790, 0.2051022, -0.2608750, 0.2045277, -0.1879182, 0.1906723
7: -0.8963747, -0.4134366, -0.8974925, -0.4134235, -0.0987353, 0.0988355
8: -6.1052723, -4.2800303, -6.1029568, -4.2699690, -0.9565740, 0.9271021
9: -4.2500935, -2.6376293, -4.2477865, -2.6288464, -0.8139812, 0.7892343

Time for backsubstitution: 6.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2359
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2360
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 2591
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 2126
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 2101
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 3050
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2120
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 2411
type: A, layer: 1, pos: 2106
type: A, layer: 1, pos: 2132
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 2590
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2345
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2179
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 3051
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2512
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 2127
type: A, layer: 1, pos: 2073
type: A, layer: 1, pos: 2620
type: A, layer: 1, pos: 2100
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 116
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2124
type: A, layer: 1, pos: 2081
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2046
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 2125
type: A, layer: 1, pos: 2518
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 2532
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2026
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2102
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 2517
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 2503
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 3469
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 2082
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 2437
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 2488
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2475
type: A, layer: 1, pos: 2677
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2160
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 2678
type: A, layer: 1, pos: 2083
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 2067
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 2985
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2955
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2045
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 2537
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 2672
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 2341
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 2322
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 2181
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2657
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 3112
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 2611
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2553
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 2647
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 2587
type: A, layer: 1, pos: 2226
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 2225
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2129
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2241
type: A, layer: 1, pos: 2242
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2399
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2609
type: A, layer: 1, pos: 2624
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2687
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2693
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 3138
type: A, layer: 1, pos: 3139
type: A, layer: 1, pos: 3144

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 2359

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0376129, upper bound: 0.0375932
time: 8.04 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0376381, upper bound: 0.0375742
time: 117.84 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -4.7792149, -2.9990203, -4.7790623, -2.9990709, -1.3680389, 1.3668375
1: -4.8729300, -2.3032351, -4.8713851, -2.3032951, -1.7877223, 1.7852902
2: -0.5827920, -0.0907726, -0.5828366, -0.0905474, -0.4155705, 0.4110142
3: -0.5534723, -0.0487759, -0.5536547, -0.0489124, -0.2201889, 0.2214017
4: -0.6760597, 0.1212651, -0.6776934, 0.1208705, -0.7721126, 0.7767223
5: -0.8906003, -0.2093980, -0.8910238, -0.2099087, -0.2605816, 0.2628796
6: -0.2614689, 0.2051568, -0.2616981, 0.2056574, -0.1896246, 0.1905009
7: -0.8963860, -0.4137366, -0.8963305, -0.4138196, -0.0986758, 0.0963622
8: -6.1059837, -4.2799969, -6.1037478, -4.2792268, -0.9325600, 0.9277484
9: -4.2489486, -2.6375880, -4.2464333, -2.6373489, -0.7948618, 0.7889466

Time for backsubstitution: 7.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2359
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2360
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 2591
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 2126
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 2101
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 3050
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2120
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 2411
type: A, layer: 1, pos: 2106
type: A, layer: 1, pos: 2132
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 2590
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2345
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2179
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 3051
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2512
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 2127
type: A, layer: 1, pos: 2073
type: A, layer: 1, pos: 2620
type: A, layer: 1, pos: 2100
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 116
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2124
type: A, layer: 1, pos: 2081
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2046
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 2125
type: A, layer: 1, pos: 2518
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 2532
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2026
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2102
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 2517
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 2503
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 2082
type: A, layer: 1, pos: 3469
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 2437
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 2488
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2475
type: A, layer: 1, pos: 2677
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2160
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 2678
type: A, layer: 1, pos: 2083
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 2067
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 2985
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2955
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2045
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 2537
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 2672
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 2341
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 2322
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 2181
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2657
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 3112
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 2611
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2553
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 2647
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 2587
type: A, layer: 1, pos: 2226
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 2225
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2129
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2241
type: A, layer: 1, pos: 2242
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2399
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2609
type: A, layer: 1, pos: 2624
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2687
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2693
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 3138
type: A, layer: 1, pos: 3139
type: A, layer: 1, pos: 3144

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 2359

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0375814, upper bound: 0.0376466
time: 141.78 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0376050, upper bound: 0.0376424
time: 139.43 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4.7792897, -2.9990206, -4.7794399, -2.9895418, -1.3832469, 1.3672082
1: -4.8754263, -2.3032351, -4.8749228, -2.2884312, -1.8151922, 1.7872047
2: -0.5827930, -0.0907215, -0.5835176, -0.0904844, -0.4156110, 0.4119742
3: -0.5534707, -0.0482750, -0.5557408, -0.0482568, -0.2203534, 0.2254214
4: -0.6760661, 0.1215310, -0.6785695, 0.1211827, -0.7722687, 0.7779617
5: -0.8906004, -0.2087543, -0.8939486, -0.2090938, -0.2607244, 0.2690547
6: -0.2614626, 0.2051028, -0.2619110, 0.2055972, -0.1896293, 0.1911435
7: -0.8963861, -0.4133869, -0.8980553, -0.4133653, -0.0987578, 0.0998861
8: -6.1060271, -4.2799945, -6.1038809, -4.2691154, -0.9582348, 0.9281244
9: -4.2502079, -2.6375887, -4.2479134, -2.6285472, -0.8152790, 0.7893081

Time for backsubstitution: 7.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2359
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2360
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 2591
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2552
type: A, layer: 1, pos: 2126
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 2101
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 3050
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2120
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 2411
type: A, layer: 1, pos: 2106
type: A, layer: 1, pos: 2132
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 2590
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2345
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2179
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 3051
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2512
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 2127
type: A, layer: 1, pos: 2073
type: A, layer: 1, pos: 2620
type: A, layer: 1, pos: 2100
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 116
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2124
type: A, layer: 1, pos: 2081
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2046
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 2125
type: A, layer: 1, pos: 2518
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 2532
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2026
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 2128
type: A, layer: 1, pos: 2102
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 2517
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 2503
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 2082
type: A, layer: 1, pos: 3469
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 2437
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 2488
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2475
type: A, layer: 1, pos: 2677
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2160
type: A, layer: 1, pos: 2377
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 2678
type: A, layer: 1, pos: 2083
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 2067
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 2985
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2955
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2045
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 2537
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 2672
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 2341
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 2322
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 2181
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 2632
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 2180
type: A, layer: 1, pos: 2657
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 3112
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 2611
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 2052
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2553
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 2647
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 2587
type: A, layer: 1, pos: 2226
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 2225
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2129
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2241
type: A, layer: 1, pos: 2242
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2399
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2609
type: A, layer: 1, pos: 2624
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2687
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2693
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 3138
type: A, layer: 1, pos: 3139
type: A, layer: 1, pos: 3144

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2359

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376151, upper bound: 0.0376610
time: 6.36 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376380, upper bound: 0.0376530
time: 54.42 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 68.12 seconds
NS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 68.12
Output dim: 2, lower bound: -0.0376124, upper bound: 0.0374323
NS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 68.12
Output dim: 2, lower bound: -0.0376361, upper bound: 0.0374353
NS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 68.12
Output dim: 2, lower bound: -0.0376137, upper bound: 0.0374855
NS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 68.12
Output dim: 2, lower bound: -0.0376390, upper bound: 0.0374202
NS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 68.12
Output dim: 2, lower bound: -0.0376129, upper bound: 0.0375932
NS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 68.12
Output dim: 2, lower bound: -0.0376381, upper bound: 0.0375742
NS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 68.12
Output dim: 2, lower bound: -0.0375814, upper bound: 0.0376466
NS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 68.12
Output dim: 2, lower bound: -0.0376050, upper bound: 0.0376424
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 68.12
Output dim: 2, lower bound: -0.0376151, upper bound: 0.0376610
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 68.12
Output dim: 2, lower bound: -0.0376380, upper bound: 0.0376530

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4.7746630, -3.0139942, -4.7787681, -3.0030684, -1.3539557, 1.3466842
1: -4.8659811, -2.3288479, -4.8744402, -2.3110752, -1.7700262, 1.7558088
2: -0.5821383, -0.0907274, -0.5829604, -0.0904861, -0.4149469, 0.4113315
3: -0.5479856, -0.0503632, -0.5508911, -0.0483215, -0.2131182, 0.2146946
4: -0.6747588, 0.1208986, -0.6773879, 0.1211760, -0.7714630, 0.7774888
5: -0.8841393, -0.2111685, -0.8882225, -0.2090998, -0.2518978, 0.2558408
6: -0.2603549, 0.2051779, -0.2609043, 0.2055939, -0.1885756, 0.1896359
7: -0.8952771, -0.4135744, -0.8970637, -0.4133719, -0.0967644, 0.0965856
8: -6.1045275, -4.2883940, -6.1038809, -4.2766204, -0.9387758, 0.9148178
9: -4.2464056, -2.6493757, -4.2478981, -2.6390312, -0.7909738, 0.7728627

Time for backsubstitution: 7.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2360
type: B, layer: 1, pos: 2600
type: B, layer: 1, pos: 2591
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 2126
type: B, layer: 1, pos: 2087
type: B, layer: 1, pos: 2101
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 3050
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2120
type: B, layer: 1, pos: 335
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 2411
type: B, layer: 1, pos: 2106
type: B, layer: 1, pos: 2132
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2590
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 2539
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2179
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 3051
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 2127
type: B, layer: 1, pos: 2073
type: B, layer: 1, pos: 2620
type: B, layer: 1, pos: 2100
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 116
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2124
type: B, layer: 1, pos: 2081
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2046
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 2518
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 2532
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 2617
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2026
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2102
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 2517
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 145
type: B, layer: 1, pos: 2503
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 2082
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 2437
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 2488
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 2091
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2475
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2160
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 2678
type: B, layer: 1, pos: 2083
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 2067
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 2203
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 2985
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 2666
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2955
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2045
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 2537
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 2672
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 2341
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 2322
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 2181
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2657
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 3112
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 2547
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 2231
type: B, layer: 1, pos: 2611
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 2052
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2074
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2553
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 2647
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 2226
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 2225
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2129
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2234
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2241
type: B, layer: 1, pos: 2242
type: B, layer: 1, pos: 2243
type: B, layer: 1, pos: 2399
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2609
type: B, layer: 1, pos: 2624
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2687
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2693
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 3138
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3144

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3086

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0376112, upper bound: 0.0376082
time: 181.56 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0376110, upper bound: 0.0376386
time: 170.23 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.7791204, -3.0019953, -4.7792873, -2.9922795, -1.3821392, 1.3406403
1: -4.8753076, -2.3076301, -4.8748198, -2.2924614, -1.8144832, 1.7499058
2: -0.5825916, -0.0907224, -0.5833397, -0.0904849, -0.4151005, 0.4116079
3: -0.5526494, -0.0482925, -0.5549974, -0.0482718, -0.2091908, 0.2253007
4: -0.6759048, 0.1215287, -0.6784306, 0.1211812, -0.7716181, 0.7774796
5: -0.8896724, -0.2087570, -0.8931066, -0.2090957, -0.2465789, 0.2689137
6: -0.2609988, 0.2051018, -0.2615016, 0.2055964, -0.1881080, 0.1905690
7: -0.8959196, -0.4133897, -0.8976275, -0.4133678, -0.0957803, 0.0998496
8: -6.1060266, -4.2816267, -6.1038814, -4.2706108, -0.9578372, 0.9070778
9: -4.2502027, -2.6395812, -4.2479095, -2.6303620, -0.8150234, 0.7633290

Time for backsubstitution: 6.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2360
type: B, layer: 1, pos: 2600
type: B, layer: 1, pos: 2591
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2552
type: B, layer: 1, pos: 2126
type: B, layer: 1, pos: 2087
type: B, layer: 1, pos: 2101
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 3050
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2120
type: B, layer: 1, pos: 335
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 2411
type: B, layer: 1, pos: 2106
type: B, layer: 1, pos: 2132
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2590
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 2539
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2345
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2179
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3051
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 2127
type: B, layer: 1, pos: 2073
type: B, layer: 1, pos: 2620
type: B, layer: 1, pos: 2100
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 116
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2124
type: B, layer: 1, pos: 2081
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2046
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 2125
type: B, layer: 1, pos: 2518
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 2532
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 2617
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2026
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 2128
type: B, layer: 1, pos: 2102
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 2517
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 145
type: B, layer: 1, pos: 2503
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 2082
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 2437
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 2488
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 2091
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2475
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2160
type: B, layer: 1, pos: 2377
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 2678
type: B, layer: 1, pos: 2083
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 2067
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 2203
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 2985
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 2666
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2955
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 2045
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 2537
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 2672
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 2341
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 2322
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 2181
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 2180
type: B, layer: 1, pos: 2657
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 3112
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 2547
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 2231
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 2611
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 2052
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2074
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2553
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 2647
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 2226
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 2225
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2129
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2234
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2241
type: B, layer: 1, pos: 2242
type: B, layer: 1, pos: 2243
type: B, layer: 1, pos: 2399
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2609
type: B, layer: 1, pos: 2624
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2687
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2693
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 3138
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3144

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3086

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0376353, upper bound: 0.0376042
time: 134.42 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0376360, upper bound: 0.0376483
time: 149.29 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 290.73 seconds
NS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 290.73
Output dim: 2, lower bound: -0.0376112, upper bound: 0.0376082
NS_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 290.73
Output dim: 2, lower bound: -0.0376110, upper bound: 0.0376386
NS_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 290.73
Output dim: 2, lower bound: -0.0376353, upper bound: 0.0376042
NS_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 290.73
Output dim: 2, lower bound: -0.0376360, upper bound: 0.0376483

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 33.41 + 3114.15 = 3147.57 seconds
