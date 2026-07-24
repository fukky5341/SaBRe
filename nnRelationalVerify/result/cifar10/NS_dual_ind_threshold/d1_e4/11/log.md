## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 11)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.0490835673


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-2.9899344, -1.6791523, -2.9899344, -1.6791523, -0.4830543, 0.4830543)
1: (-5.4404860, -3.4300203, -5.4404860, -3.4300203, -0.8298308, 0.8298308)
2: (-0.2036541, 0.1937240, -0.2036541, 0.1937240, -0.1220713, 0.1220713)
3: (-2.3926487, -1.7725695, -2.3926487, -1.7725695, -0.1881570, 0.1881570)
4: (0.4635379, 0.5799251, 0.4635379, 0.5799251, -0.0711872, 0.0711872)
5: (-2.4017086, -1.8002263, -2.4017086, -1.8002263, -0.2739069, 0.2739069)
6: (-0.2971857, 0.1779544, -0.2971857, 0.1779544, -0.3865951, 0.3865951)
7: (-1.8941863, -1.2658451, -1.8941863, -1.2658451, -0.4546693, 0.4546692)
8: (-4.2648854, -3.2627838, -4.2648854, -3.2627838, -0.4648452, 0.4648451)
9: (-5.9969206, -4.8441453, -5.9969206, -4.8441453, -0.4469083, 0.4469082)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.87 + 51.78 = 59.65 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.0491315, upper bound: 0.0491336

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 280
type: A, layer: 1, pos: 263
type: A, layer: 1, pos: 265
type: A, layer: 1, pos: 294
type: A, layer: 1, pos: 262
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 338
type: A, layer: 1, pos: 248
type: A, layer: 1, pos: 339
type: A, layer: 1, pos: 279
type: A, layer: 1, pos: 3478
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 3483
type: A, layer: 1, pos: 3255
type: A, layer: 1, pos: 354
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 3257
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 493
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 2106
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 2647
type: A, layer: 1, pos: 2876
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 2863
type: A, layer: 1, pos: 2848
type: A, layer: 1, pos: 2874
type: A, layer: 1, pos: 3459
type: A, layer: 1, pos: 2883
type: A, layer: 1, pos: 2846
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 2897
type: A, layer: 1, pos: 310
type: A, layer: 1, pos: 3396
type: A, layer: 1, pos: 2888
type: A, layer: 1, pos: 2845
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 2872
type: A, layer: 1, pos: 2873
type: A, layer: 1, pos: 2858
type: A, layer: 1, pos: 2859
type: A, layer: 1, pos: 2851
type: A, layer: 1, pos: 2853
type: A, layer: 1, pos: 2838
type: A, layer: 1, pos: 2837
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 2885
type: A, layer: 1, pos: 2902
type: A, layer: 1, pos: 2831
type: A, layer: 1, pos: 2889
type: A, layer: 1, pos: 2865
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 3447
type: A, layer: 1, pos: 98
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2850
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2121
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2881
type: A, layer: 1, pos: 2887
type: A, layer: 1, pos: 2820
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2854
type: A, layer: 1, pos: 3306
type: A, layer: 1, pos: 2905
type: A, layer: 1, pos: 2870
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 2900
type: A, layer: 1, pos: 2122
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2877
type: A, layer: 1, pos: 3263
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 3432
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 3067
type: A, layer: 1, pos: 2904
type: A, layer: 1, pos: 2880
type: A, layer: 1, pos: 2120
type: A, layer: 1, pos: 2572
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 3489
type: A, layer: 1, pos: 2057
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 2855
type: A, layer: 1, pos: 2852
type: A, layer: 1, pos: 2029
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2055
type: A, layer: 1, pos: 2830
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2901
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 507
type: A, layer: 1, pos: 2028
type: A, layer: 1, pos: 2884
type: A, layer: 1, pos: 2097
type: A, layer: 1, pos: 3465
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 3292
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 2070
type: A, layer: 1, pos: 2895
type: A, layer: 1, pos: 2839
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 3477
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 295
type: A, layer: 1, pos: 2906
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2841
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 2868
type: A, layer: 1, pos: 3472
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2570
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 2844
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 2083
type: A, layer: 1, pos: 2821
type: A, layer: 1, pos: 3316
type: A, layer: 1, pos: 2826
type: A, layer: 1, pos: 2571
type: A, layer: 1, pos: 3322
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 2086
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 2092
type: A, layer: 1, pos: 2123
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 2027
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 2828
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 2840
type: A, layer: 1, pos: 2496
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 3321
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 2084
type: A, layer: 1, pos: 2685
type: A, layer: 1, pos: 2849
type: A, layer: 1, pos: 2879
type: A, layer: 1, pos: 2894
type: A, layer: 1, pos: 2921
type: A, layer: 1, pos: 2923
type: A, layer: 1, pos: 2924
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3362
type: A, layer: 1, pos: 3365
type: A, layer: 1, pos: 3366
type: A, layer: 1, pos: 3369
type: A, layer: 1, pos: 3370
type: A, layer: 1, pos: 3371
type: A, layer: 1, pos: 3372

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 280

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0489191, upper bound: 0.0490960
time: 19.02 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0491057, upper bound: 0.0491064
time: 95.60 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 114.69 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 114.69
Output dim: 4, lower bound: -0.0489191, upper bound: 0.0490960
NS_A2, status: Status.UNKNOWN, split count: 1, time: 114.69
Output dim: 4, lower bound: -0.0491057, upper bound: 0.0491064

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -2.9895873, -1.6791527, -2.9896424, -1.6791527, -0.4800755, 0.4805259
1: -5.4404593, -3.4305348, -5.4404631, -3.4304538, -0.8293568, 0.8292741
2: -0.2030787, 0.1937240, -0.2031641, 0.1937240, -0.1213470, 0.1214462
3: -2.3926451, -1.7726672, -2.3926454, -1.7726611, -0.1870456, 0.1868567
4: 0.4649013, 0.5799251, 0.4646882, 0.5799251, -0.0697575, 0.0699743
5: -2.4017086, -1.8018115, -2.4017084, -1.8016617, -0.2712018, 0.2707036
6: -0.2956164, 0.1779535, -0.2958630, 0.1779537, -0.3850161, 0.3852639
7: -1.8941840, -1.2685635, -1.8941844, -1.2681787, -0.4524017, 0.4520764
8: -4.2641716, -3.2627861, -4.2642846, -3.2627852, -0.4638751, 0.4639860
9: -5.9969149, -4.8471818, -5.9969163, -4.8467202, -0.4440894, 0.4435654

Time for backsubstitution: 6.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 263
type: B, layer: 1, pos: 265
type: B, layer: 1, pos: 294
type: B, layer: 1, pos: 262
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 248
type: B, layer: 1, pos: 339
type: B, layer: 1, pos: 280
type: B, layer: 1, pos: 3478
type: B, layer: 1, pos: 279
type: B, layer: 1, pos: 3462
type: B, layer: 1, pos: 3483
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 3255
type: B, layer: 1, pos: 354
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 3257
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 493
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 2106
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 2647
type: B, layer: 1, pos: 2876
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 2863
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 2848
type: B, layer: 1, pos: 2874
type: B, layer: 1, pos: 3459
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 2846
type: B, layer: 1, pos: 2091
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 2897
type: B, layer: 1, pos: 310
type: B, layer: 1, pos: 2888
type: B, layer: 1, pos: 2845
type: B, layer: 1, pos: 82
type: B, layer: 1, pos: 3396
type: B, layer: 1, pos: 2872
type: B, layer: 1, pos: 2873
type: B, layer: 1, pos: 2858
type: B, layer: 1, pos: 2859
type: B, layer: 1, pos: 2851
type: B, layer: 1, pos: 2853
type: B, layer: 1, pos: 2838
type: B, layer: 1, pos: 2837
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2902
type: B, layer: 1, pos: 2831
type: B, layer: 1, pos: 2889
type: B, layer: 1, pos: 3447
type: B, layer: 1, pos: 2865
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 98
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2850
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 2121
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2881
type: B, layer: 1, pos: 2887
type: B, layer: 1, pos: 2820
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2854
type: B, layer: 1, pos: 3306
type: B, layer: 1, pos: 2905
type: B, layer: 1, pos: 2870
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 2900
type: B, layer: 1, pos: 2122
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2877
type: B, layer: 1, pos: 3263
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 3432
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 3067
type: B, layer: 1, pos: 2904
type: B, layer: 1, pos: 2880
type: B, layer: 1, pos: 2120
type: B, layer: 1, pos: 2572
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 2057
type: B, layer: 1, pos: 2029
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 2855
type: B, layer: 1, pos: 2852
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2055
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2830
type: B, layer: 1, pos: 2056
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2901
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 2028
type: B, layer: 1, pos: 507
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 2097
type: B, layer: 1, pos: 3465
type: B, layer: 1, pos: 2070
type: B, layer: 1, pos: 2617
type: B, layer: 1, pos: 3292
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 2895
type: B, layer: 1, pos: 2839
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 2074
type: B, layer: 1, pos: 3477
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 295
type: B, layer: 1, pos: 2906
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2841
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2868
type: B, layer: 1, pos: 3472
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 2570
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2844
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 2821
type: B, layer: 1, pos: 3316
type: B, layer: 1, pos: 2826
type: B, layer: 1, pos: 2083
type: B, layer: 1, pos: 2571
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2030
type: B, layer: 1, pos: 3322
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 2092
type: B, layer: 1, pos: 2086
type: B, layer: 1, pos: 2123
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 2027
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 2828
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 2842
type: B, layer: 1, pos: 2840
type: B, layer: 1, pos: 2496
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 3321
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 3081
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 2084
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 2849
type: B, layer: 1, pos: 2879
type: B, layer: 1, pos: 2894
type: B, layer: 1, pos: 2921
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 2924
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3362
type: B, layer: 1, pos: 3365
type: B, layer: 1, pos: 3366
type: B, layer: 1, pos: 3369
type: B, layer: 1, pos: 3370
type: B, layer: 1, pos: 3371
type: B, layer: 1, pos: 3372

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 263

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0488767, upper bound: 0.0489476
time: 87.63 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0489043, upper bound: 0.0490853
time: 28.31 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -2.9858639, -1.6767107, -2.9858575, -1.6791527, -0.4808878, 0.4919554
1: -5.4428053, -3.4299688, -5.4404793, -3.4300532, -0.8321853, 0.8294696
2: -0.2037390, 0.1951431, -0.2034370, 0.1937240, -0.1217479, 0.1234688
3: -2.3945799, -1.7742591, -2.3926473, -1.7745504, -0.1936200, 0.1873669
4: 0.4609860, 0.5841655, 0.4636543, 0.5799243, -0.0730440, 0.0753438
5: -2.4055328, -1.8004940, -2.4017019, -1.8015950, -0.2811733, 0.2729749
6: -0.2971718, 0.1845451, -0.2970813, 0.1779246, -0.3855615, 0.3931041
7: -1.9065776, -1.2651021, -1.8941076, -1.2661620, -0.4660701, 0.4546243
8: -4.2649660, -3.2631807, -4.2629013, -3.2628183, -0.4680538, 0.4662283
9: -6.0085850, -4.8440862, -5.9969196, -4.8443937, -0.4591526, 0.4445370

Time for backsubstitution: 6.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 263
type: B, layer: 1, pos: 265
type: B, layer: 1, pos: 294
type: B, layer: 1, pos: 262
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 248
type: B, layer: 1, pos: 339
type: B, layer: 1, pos: 280
type: B, layer: 1, pos: 279
type: B, layer: 1, pos: 3478
type: B, layer: 1, pos: 3462
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 3483
type: B, layer: 1, pos: 3255
type: B, layer: 1, pos: 354
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 3257
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 493
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 2106
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 2647
type: B, layer: 1, pos: 2876
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 2863
type: B, layer: 1, pos: 2848
type: B, layer: 1, pos: 2874
type: B, layer: 1, pos: 3459
type: B, layer: 1, pos: 2846
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 2091
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 2897
type: B, layer: 1, pos: 3396
type: B, layer: 1, pos: 310
type: B, layer: 1, pos: 2888
type: B, layer: 1, pos: 2845
type: B, layer: 1, pos: 82
type: B, layer: 1, pos: 2872
type: B, layer: 1, pos: 2873
type: B, layer: 1, pos: 2858
type: B, layer: 1, pos: 2859
type: B, layer: 1, pos: 2851
type: B, layer: 1, pos: 2853
type: B, layer: 1, pos: 2838
type: B, layer: 1, pos: 2837
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2902
type: B, layer: 1, pos: 2831
type: B, layer: 1, pos: 2889
type: B, layer: 1, pos: 2865
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 3447
type: B, layer: 1, pos: 98
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2850
type: B, layer: 1, pos: 2847
type: B, layer: 1, pos: 2121
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2881
type: B, layer: 1, pos: 2887
type: B, layer: 1, pos: 2820
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 2854
type: B, layer: 1, pos: 3306
type: B, layer: 1, pos: 2905
type: B, layer: 1, pos: 2870
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 2122
type: B, layer: 1, pos: 2900
type: B, layer: 1, pos: 2877
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 3263
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 3432
type: B, layer: 1, pos: 2904
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 2120
type: B, layer: 1, pos: 3067
type: B, layer: 1, pos: 2880
type: B, layer: 1, pos: 2572
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 2057
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 2855
type: B, layer: 1, pos: 2852
type: B, layer: 1, pos: 2029
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2830
type: B, layer: 1, pos: 2055
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 3066
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 2901
type: B, layer: 1, pos: 2056
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 507
type: B, layer: 1, pos: 2028
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 2097
type: B, layer: 1, pos: 3465
type: B, layer: 1, pos: 2617
type: B, layer: 1, pos: 3292
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 2070
type: B, layer: 1, pos: 2895
type: B, layer: 1, pos: 2839
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 2074
type: B, layer: 1, pos: 3477
type: B, layer: 1, pos: 3349
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 295
type: B, layer: 1, pos: 2906
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2841
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2868
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 2570
type: B, layer: 1, pos: 3472
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 2844
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 2083
type: B, layer: 1, pos: 2821
type: B, layer: 1, pos: 2826
type: B, layer: 1, pos: 3316
type: B, layer: 1, pos: 2571
type: B, layer: 1, pos: 3322
type: B, layer: 1, pos: 2030
type: B, layer: 1, pos: 2086
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 2092
type: B, layer: 1, pos: 2123
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 2828
type: B, layer: 1, pos: 2027
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 2842
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 2840
type: B, layer: 1, pos: 2496
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 3321
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 3081
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 2084
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 2849
type: B, layer: 1, pos: 2879
type: B, layer: 1, pos: 2894
type: B, layer: 1, pos: 2921
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 2924
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3362
type: B, layer: 1, pos: 3365
type: B, layer: 1, pos: 3366
type: B, layer: 1, pos: 3369
type: B, layer: 1, pos: 3370
type: B, layer: 1, pos: 3371
type: B, layer: 1, pos: 3372

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 263

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0490639, upper bound: 0.0489561
time: 186.99 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0490867, upper bound: 0.0490901
time: 41.64 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 234.73 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 234.73
Output dim: 4, lower bound: -0.0488767, upper bound: 0.0489476
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 234.73
Output dim: 4, lower bound: -0.0489043, upper bound: 0.0490853
NS_A2_B1, status: Status.VERIFIED, split count: 2, time: 234.73
Output dim: 4, lower bound: -0.0490639, upper bound: 0.0489561
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 234.73
Output dim: 4, lower bound: -0.0490867, upper bound: 0.0490901

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -2.9895868, -1.6791735, -2.9932697, -1.6791750, -0.4778861, 0.4831011
1: -5.4391651, -3.4305344, -5.4390965, -3.4308801, -0.8308417, 0.8281542
2: -0.2030785, 0.1936942, -0.2074118, 0.1936908, -0.1184864, 0.1261745
3: -2.3926444, -1.7731593, -2.3941522, -1.7728627, -0.1855228, 0.1900069
4: 0.4649371, 0.5799245, 0.4625605, 0.5816195, -0.0714191, 0.0713051
5: -2.4017026, -1.8018892, -2.4083407, -1.8008239, -0.2675410, 0.2779431
6: -0.2955443, 0.1779489, -0.2957882, 0.1835634, -0.3903880, 0.3819703
7: -1.8941672, -1.2686642, -1.9042351, -1.2679645, -0.4461846, 0.4614295
8: -4.2636337, -3.2627881, -4.2653108, -3.2631423, -0.4644782, 0.4662257
9: -5.9969158, -4.8472509, -6.0004034, -4.8465261, -0.4422770, 0.4468137

Time for backsubstitution: 6.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 265
type: A, layer: 1, pos: 294
type: A, layer: 1, pos: 262
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 338
type: A, layer: 1, pos: 248
type: A, layer: 1, pos: 339
type: A, layer: 1, pos: 3478
type: A, layer: 1, pos: 279
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 3483
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 263
type: A, layer: 1, pos: 3255
type: A, layer: 1, pos: 354
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 3257
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 493
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 2106
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 2647
type: A, layer: 1, pos: 2876
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 2863
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 2848
type: A, layer: 1, pos: 2874
type: A, layer: 1, pos: 3459
type: A, layer: 1, pos: 2883
type: A, layer: 1, pos: 2846
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 2897
type: A, layer: 1, pos: 310
type: A, layer: 1, pos: 2888
type: A, layer: 1, pos: 2845
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 2872
type: A, layer: 1, pos: 3396
type: A, layer: 1, pos: 2873
type: A, layer: 1, pos: 2858
type: A, layer: 1, pos: 2859
type: A, layer: 1, pos: 2851
type: A, layer: 1, pos: 2853
type: A, layer: 1, pos: 2838
type: A, layer: 1, pos: 2837
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 2885
type: A, layer: 1, pos: 2902
type: A, layer: 1, pos: 2831
type: A, layer: 1, pos: 2889
type: A, layer: 1, pos: 3447
type: A, layer: 1, pos: 2865
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 98
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2850
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2121
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2881
type: A, layer: 1, pos: 2887
type: A, layer: 1, pos: 2820
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2854
type: A, layer: 1, pos: 3306
type: A, layer: 1, pos: 2905
type: A, layer: 1, pos: 2870
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2122
type: A, layer: 1, pos: 2900
type: A, layer: 1, pos: 2877
type: A, layer: 1, pos: 3263
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 3432
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 3067
type: A, layer: 1, pos: 2904
type: A, layer: 1, pos: 2120
type: A, layer: 1, pos: 2880
type: A, layer: 1, pos: 2572
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 3489
type: A, layer: 1, pos: 2057
type: A, layer: 1, pos: 2029
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 2855
type: A, layer: 1, pos: 2852
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2055
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 2830
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2901
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 2028
type: A, layer: 1, pos: 507
type: A, layer: 1, pos: 2884
type: A, layer: 1, pos: 2097
type: A, layer: 1, pos: 3465
type: A, layer: 1, pos: 2070
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 3292
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 2895
type: A, layer: 1, pos: 2839
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 3477
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 295
type: A, layer: 1, pos: 2906
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2841
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 2868
type: A, layer: 1, pos: 3472
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2570
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 2844
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 2821
type: A, layer: 1, pos: 3316
type: A, layer: 1, pos: 2826
type: A, layer: 1, pos: 2083
type: A, layer: 1, pos: 2571
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 3322
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 2092
type: A, layer: 1, pos: 2086
type: A, layer: 1, pos: 2123
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 2027
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 2828
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 2840
type: A, layer: 1, pos: 2496
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 3321
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 2084
type: A, layer: 1, pos: 2685
type: A, layer: 1, pos: 2849
type: A, layer: 1, pos: 2879
type: A, layer: 1, pos: 2894
type: A, layer: 1, pos: 2921
type: A, layer: 1, pos: 2923
type: A, layer: 1, pos: 2924
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3362
type: A, layer: 1, pos: 3365
type: A, layer: 1, pos: 3366
type: A, layer: 1, pos: 3369
type: A, layer: 1, pos: 3370
type: A, layer: 1, pos: 3371
type: A, layer: 1, pos: 3372

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 265

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0487475, upper bound: 0.0490577
time: 51.99 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0488899, upper bound: 0.0490795
time: 182.88 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -2.9858639, -1.6767313, -2.9894788, -1.6791755, -0.4786981, 0.4944813
1: -5.4415126, -3.4299693, -5.4391108, -3.4304795, -0.8336654, 0.8283502
2: -0.2037387, 0.1951132, -0.2076744, 0.1936908, -0.1188873, 0.1281711
3: -2.3945792, -1.7747517, -2.3941536, -1.7747529, -0.1920974, 0.1905160
4: 0.4610211, 0.5841648, 0.4615292, 0.5816189, -0.0747058, 0.0766675
5: -2.4055271, -1.8005764, -2.4083338, -1.8007591, -0.2774904, 0.2802103
6: -0.2970998, 0.1845404, -0.2970065, 0.1835344, -0.3909333, 0.3898105
7: -1.9065601, -1.2652031, -1.9041553, -1.2659439, -0.4598573, 0.4639746
8: -4.2644067, -3.2631817, -4.2639637, -3.2631757, -0.4686465, 0.4684513
9: -6.0085850, -4.8441544, -6.0004063, -4.8441954, -0.4573417, 0.4477854

Time for backsubstitution: 6.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 294
type: A, layer: 1, pos: 265
type: A, layer: 1, pos: 262
type: A, layer: 1, pos: 338
type: A, layer: 1, pos: 339
type: A, layer: 1, pos: 248
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 3478
type: A, layer: 1, pos: 279
type: A, layer: 1, pos: 3483
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 263
type: A, layer: 1, pos: 3255
type: A, layer: 1, pos: 354
type: A, layer: 1, pos: 493
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 3257
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 2106
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 2647
type: A, layer: 1, pos: 2876
type: A, layer: 1, pos: 3447
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 3459
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 2863
type: A, layer: 1, pos: 2848
type: A, layer: 1, pos: 2874
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 2846
type: A, layer: 1, pos: 2883
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 2897
type: A, layer: 1, pos: 2888
type: A, layer: 1, pos: 310
type: A, layer: 1, pos: 2845
type: A, layer: 1, pos: 82
type: A, layer: 1, pos: 2872
type: A, layer: 1, pos: 2873
type: A, layer: 1, pos: 2858
type: A, layer: 1, pos: 2859
type: A, layer: 1, pos: 2851
type: A, layer: 1, pos: 3396
type: A, layer: 1, pos: 2853
type: A, layer: 1, pos: 2838
type: A, layer: 1, pos: 2837
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 2885
type: A, layer: 1, pos: 2902
type: A, layer: 1, pos: 2831
type: A, layer: 1, pos: 2889
type: A, layer: 1, pos: 2865
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 98
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 2847
type: A, layer: 1, pos: 2850
type: A, layer: 1, pos: 2121
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2881
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 2887
type: A, layer: 1, pos: 2820
type: A, layer: 1, pos: 3306
type: A, layer: 1, pos: 2854
type: A, layer: 1, pos: 2905
type: A, layer: 1, pos: 2870
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 3432
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2122
type: A, layer: 1, pos: 2900
type: A, layer: 1, pos: 2877
type: A, layer: 1, pos: 3263
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 2904
type: A, layer: 1, pos: 2120
type: A, layer: 1, pos: 3067
type: A, layer: 1, pos: 2880
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 2572
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 2057
type: A, layer: 1, pos: 2029
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 2852
type: A, layer: 1, pos: 2855
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2055
type: A, layer: 1, pos: 2830
type: A, layer: 1, pos: 3066
type: A, layer: 1, pos: 507
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 2901
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 3489
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 2884
type: A, layer: 1, pos: 2028
type: A, layer: 1, pos: 3465
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 2097
type: A, layer: 1, pos: 2070
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 3292
type: A, layer: 1, pos: 2895
type: A, layer: 1, pos: 2839
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 3477
type: A, layer: 1, pos: 3349
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 295
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 2906
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2841
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 2868
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2570
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 2844
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 2821
type: A, layer: 1, pos: 2826
type: A, layer: 1, pos: 3316
type: A, layer: 1, pos: 2571
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 3322
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 2092
type: A, layer: 1, pos: 2123
type: A, layer: 1, pos: 2086
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 2828
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 2027
type: A, layer: 1, pos: 2083
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 3472
type: A, layer: 1, pos: 2840
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 2496
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 3321
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 2084
type: A, layer: 1, pos: 2685
type: A, layer: 1, pos: 2849
type: A, layer: 1, pos: 2879
type: A, layer: 1, pos: 2894
type: A, layer: 1, pos: 2921
type: A, layer: 1, pos: 2923
type: A, layer: 1, pos: 2924
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3362
type: A, layer: 1, pos: 3365
type: A, layer: 1, pos: 3366
type: A, layer: 1, pos: 3369
type: A, layer: 1, pos: 3370
type: A, layer: 1, pos: 3371
type: A, layer: 1, pos: 3372

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 294

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0489757, upper bound: 0.0490445
time: 96.12 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0490418, upper bound: 0.0490433
time: 177.00 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 279.26 seconds
NS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 279.26
Output dim: 4, lower bound: -0.0487475, upper bound: 0.0490577
NS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 279.26
Output dim: 4, lower bound: -0.0488899, upper bound: 0.0490795
NS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 279.26
Output dim: 4, lower bound: -0.0489757, upper bound: 0.0490445
NS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 279.26
Output dim: 4, lower bound: -0.0490418, upper bound: 0.0490433

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 59.65 + 991.70 = 1051.34 seconds
