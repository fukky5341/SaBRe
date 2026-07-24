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
execution time: IAR + RelationalAnalysis = 7.71 + 51.23 = 58.93 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.0491315, upper bound: 0.0491336

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 562
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 577
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 2854
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2859
type: DSZ, layer: 1, pos: 2874
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2901
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2900
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2902
type: DSZ, layer: 1, pos: 2845
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 2837
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2905
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 280
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 265
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 279
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 248
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 2821
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 310
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 3322
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 507
type: DSZ, layer: 1, pos: 3396
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 294
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 591
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 519
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2880
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2895
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2924
type: DSZ, layer: 1, pos: 3255
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3365
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3370
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3465
type: DSZ, layer: 1, pos: 3483

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 339

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0491185, upper bound: 0.0491281
time: 69.84 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0491304, upper bound: 0.0491229
time: 129.95 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 199.86 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 199.86
Output dim: 4, lower bound: -0.0491185, upper bound: 0.0491281
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 199.86
Output dim: 4, lower bound: -0.0491304, upper bound: 0.0491229

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -2.9899344, -1.6791523, -2.9899344, -1.6791523, -0.4821992, 0.4821587
1: -5.4404860, -3.4300203, -5.4404860, -3.4300203, -0.8284522, 0.8283877
2: -0.2036541, 0.1937240, -0.2036541, 0.1937240, -0.1209614, 0.1210197
3: -2.3926487, -1.7725695, -2.3926487, -1.7725695, -0.1879053, 0.1879205
4: 0.4635379, 0.5799251, 0.4635379, 0.5799251, -0.0711823, 0.0711827
5: -2.4017086, -1.8002263, -2.4017086, -1.8002263, -0.2732860, 0.2732853
6: -0.2971857, 0.1779544, -0.2971857, 0.1779544, -0.3863710, 0.3863910
7: -1.8941863, -1.2658451, -1.8941863, -1.2658451, -0.4536988, 0.4535742
8: -4.2648854, -3.2627838, -4.2648854, -3.2627838, -0.4648019, 0.4647965
9: -5.9969206, -4.8441453, -5.9969206, -4.8441453, -0.4469004, 0.4469000

Time for backsubstitution: 5.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 562
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 577
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 2854
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2859
type: DSZ, layer: 1, pos: 2874
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2901
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2900
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2902
type: DSZ, layer: 1, pos: 2845
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 2837
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2905
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 280
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 265
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 279
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 248
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 2821
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 310
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 3322
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 507
type: DSZ, layer: 1, pos: 3396
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 294
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 591
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 519
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2880
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2895
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2924
type: DSZ, layer: 1, pos: 3255
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3365
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3370
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3465
type: DSZ, layer: 1, pos: 3483

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3489

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0491194, upper bound: 0.0491330
time: 82.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0491184, upper bound: 0.0491302
time: 51.73 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -2.9899344, -1.6791523, -2.9899344, -1.6791523, -0.4821587, 0.4821992
1: -5.4404860, -3.4300203, -5.4404860, -3.4300203, -0.8283878, 0.8284522
2: -0.2036541, 0.1937240, -0.2036541, 0.1937240, -0.1210197, 0.1209614
3: -2.3926487, -1.7725695, -2.3926487, -1.7725695, -0.1879205, 0.1879053
4: 0.4635379, 0.5799251, 0.4635379, 0.5799251, -0.0711827, 0.0711823
5: -2.4017086, -1.8002263, -2.4017086, -1.8002263, -0.2732854, 0.2732860
6: -0.2971857, 0.1779544, -0.2971857, 0.1779544, -0.3863910, 0.3863711
7: -1.8941863, -1.2658451, -1.8941863, -1.2658451, -0.4535742, 0.4536988
8: -4.2648854, -3.2627838, -4.2648854, -3.2627838, -0.4647966, 0.4648019
9: -5.9969206, -4.8441453, -5.9969206, -4.8441453, -0.4469000, 0.4469003

Time for backsubstitution: 5.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 562
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 577
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 2854
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2859
type: DSZ, layer: 1, pos: 2874
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2901
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2900
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2902
type: DSZ, layer: 1, pos: 2845
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 2837
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2905
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 280
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 265
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 279
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 248
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 2821
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 310
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 3322
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 507
type: DSZ, layer: 1, pos: 3396
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 294
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 591
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 519
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2880
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2895
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2924
type: DSZ, layer: 1, pos: 3255
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3365
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3370
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3465
type: DSZ, layer: 1, pos: 3483

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3489

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0491297, upper bound: 0.0491192
time: 70.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0491306, upper bound: 0.0491205
time: 69.67 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 146.17 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 146.17
Output dim: 4, lower bound: -0.0491194, upper bound: 0.0491330
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 146.17
Output dim: 4, lower bound: -0.0491184, upper bound: 0.0491302
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 146.17
Output dim: 4, lower bound: -0.0491297, upper bound: 0.0491192
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 146.17
Output dim: 4, lower bound: -0.0491306, upper bound: 0.0491205

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.9899344, -1.6791523, -2.9899344, -1.6791523, -0.4808151, 0.4807919
1: -5.4404860, -3.4300203, -5.4404860, -3.4300203, -0.8269435, 0.8269010
2: -0.2036541, 0.1937240, -0.2036541, 0.1937240, -0.1201713, 0.1202204
3: -2.3926487, -1.7725695, -2.3926487, -1.7725695, -0.1879413, 0.1879351
4: 0.4635379, 0.5799251, 0.4635379, 0.5799251, -0.0712053, 0.0712057
5: -2.4017086, -1.8002263, -2.4017086, -1.8002263, -0.2723838, 0.2723611
6: -0.2971857, 0.1779544, -0.2971857, 0.1779544, -0.3864510, 0.3864707
7: -1.8941863, -1.2658451, -1.8941863, -1.2658451, -0.4500475, 0.4498453
8: -4.2648854, -3.2627838, -4.2648854, -3.2627838, -0.4638090, 0.4638225
9: -5.9969206, -4.8441453, -5.9969206, -4.8441453, -0.4465069, 0.4465001

Time for backsubstitution: 5.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 562
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 577
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 2854
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2859
type: DSZ, layer: 1, pos: 2874
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2901
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2900
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2902
type: DSZ, layer: 1, pos: 2845
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 2837
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2905
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 280
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 265
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 279
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 248
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 2821
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 310
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 3322
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 507
type: DSZ, layer: 1, pos: 3396
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 294
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 591
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 519
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2880
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2895
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2924
type: DSZ, layer: 1, pos: 3255
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3365
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3370
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3465
type: DSZ, layer: 1, pos: 3483

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3081

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0490967, upper bound: 0.0491082
time: 57.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0490977, upper bound: 0.0491114
time: 146.37 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.9899344, -1.6791523, -2.9899344, -1.6791523, -0.4808327, 0.4807745
1: -5.4404860, -3.4300203, -5.4404860, -3.4300203, -0.8269655, 0.8268790
2: -0.2036541, 0.1937240, -0.2036541, 0.1937240, -0.1201620, 0.1202313
3: -2.3926487, -1.7725695, -2.3926487, -1.7725695, -0.1879200, 0.1879566
4: 0.4635379, 0.5799251, 0.4635379, 0.5799251, -0.0712053, 0.0712057
5: -2.4017086, -1.8002263, -2.4017086, -1.8002263, -0.2723618, 0.2723838
6: -0.2971857, 0.1779544, -0.2971857, 0.1779544, -0.3864507, 0.3864710
7: -1.8941863, -1.2658451, -1.8941863, -1.2658451, -0.4499698, 0.4499356
8: -4.2648854, -3.2627838, -4.2648854, -3.2627838, -0.4638285, 0.4638038
9: -5.9969206, -4.8441453, -5.9969206, -4.8441453, -0.4465005, 0.4465064

Time for backsubstitution: 6.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 562
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 577
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 2854
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2859
type: DSZ, layer: 1, pos: 2874
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2901
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2900
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2902
type: DSZ, layer: 1, pos: 2845
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 2837
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2905
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 280
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 265
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 279
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 248
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 2821
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 310
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 3322
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 507
type: DSZ, layer: 1, pos: 3396
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 294
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 591
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 519
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2880
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2895
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2924
type: DSZ, layer: 1, pos: 3255
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3365
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3370
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3465
type: DSZ, layer: 1, pos: 3483

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3081

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0490983, upper bound: 0.0491112
time: 170.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0490989, upper bound: 0.0491124
time: 22.09 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.9899344, -1.6791523, -2.9899344, -1.6791523, -0.4807746, 0.4808326
1: -5.4404860, -3.4300203, -5.4404860, -3.4300203, -0.8268791, 0.8269656
2: -0.2036541, 0.1937240, -0.2036541, 0.1937240, -0.1202313, 0.1201620
3: -2.3926487, -1.7725695, -2.3926487, -1.7725695, -0.1879566, 0.1879200
4: 0.4635379, 0.5799251, 0.4635379, 0.5799251, -0.0712057, 0.0712053
5: -2.4017086, -1.8002263, -2.4017086, -1.8002263, -0.2723838, 0.2723618
6: -0.2971857, 0.1779544, -0.2971857, 0.1779544, -0.3864710, 0.3864508
7: -1.8941863, -1.2658451, -1.8941863, -1.2658451, -0.4499356, 0.4499698
8: -4.2648854, -3.2627838, -4.2648854, -3.2627838, -0.4638037, 0.4638286
9: -5.9969206, -4.8441453, -5.9969206, -4.8441453, -0.4465064, 0.4465005

Time for backsubstitution: 6.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 562
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 577
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 2854
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2859
type: DSZ, layer: 1, pos: 2874
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2901
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2900
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2902
type: DSZ, layer: 1, pos: 2845
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 2837
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2905
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 280
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 265
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 279
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 248
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 2821
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 310
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 3322
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 507
type: DSZ, layer: 1, pos: 3396
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 294
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 591
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 519
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2880
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2895
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2924
type: DSZ, layer: 1, pos: 3255
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3365
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3370
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3465
type: DSZ, layer: 1, pos: 3483

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3081

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0491096, upper bound: 0.0490978
time: 111.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0491082, upper bound: 0.0490981
time: 162.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.9899344, -1.6791523, -2.9899344, -1.6791523, -0.4807920, 0.4808151
1: -5.4404860, -3.4300203, -5.4404860, -3.4300203, -0.8269010, 0.8269436
2: -0.2036541, 0.1937240, -0.2036541, 0.1937240, -0.1202204, 0.1201713
3: -2.3926487, -1.7725695, -2.3926487, -1.7725695, -0.1879351, 0.1879413
4: 0.4635379, 0.5799251, 0.4635379, 0.5799251, -0.0712057, 0.0712053
5: -2.4017086, -1.8002263, -2.4017086, -1.8002263, -0.2723611, 0.2723838
6: -0.2971857, 0.1779544, -0.2971857, 0.1779544, -0.3864707, 0.3864509
7: -1.8941863, -1.2658451, -1.8941863, -1.2658451, -0.4498453, 0.4500476
8: -4.2648854, -3.2627838, -4.2648854, -3.2627838, -0.4638225, 0.4638091
9: -5.9969206, -4.8441453, -5.9969206, -4.8441453, -0.4465001, 0.4465069

Time for backsubstitution: 6.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 562
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 577
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 2854
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2859
type: DSZ, layer: 1, pos: 2874
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2901
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2900
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2902
type: DSZ, layer: 1, pos: 2845
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 2837
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2905
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 280
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 265
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 279
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 248
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 2821
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 310
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 3322
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 507
type: DSZ, layer: 1, pos: 3396
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 294
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 591
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 519
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2880
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2895
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2924
type: DSZ, layer: 1, pos: 3255
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3365
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3370
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3465
type: DSZ, layer: 1, pos: 3483

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3081

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0491091, upper bound: 0.0491004
time: 97.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0491081, upper bound: 0.0491009
time: 24.65 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 128.81 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 128.81
Output dim: 4, lower bound: -0.0490967, upper bound: 0.0491082
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 128.81
Output dim: 4, lower bound: -0.0490977, upper bound: 0.0491114
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 128.81
Output dim: 4, lower bound: -0.0490983, upper bound: 0.0491112
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 128.81
Output dim: 4, lower bound: -0.0490989, upper bound: 0.0491124
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 128.81
Output dim: 4, lower bound: -0.0491096, upper bound: 0.0490978
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 128.81
Output dim: 4, lower bound: -0.0491082, upper bound: 0.0490981
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 128.81
Output dim: 4, lower bound: -0.0491091, upper bound: 0.0491004
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 128.81
Output dim: 4, lower bound: -0.0491081, upper bound: 0.0491009

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.9899344, -1.6791523, -2.9899344, -1.6791523, -0.4808099, 0.4807858
1: -5.4404860, -3.4300203, -5.4404860, -3.4300203, -0.8269315, 0.8268871
2: -0.2036541, 0.1937240, -0.2036541, 0.1937240, -0.1201671, 0.1202156
3: -2.3926487, -1.7725695, -2.3926487, -1.7725695, -0.1879318, 0.1879248
4: 0.4635379, 0.5799251, 0.4635379, 0.5799251, -0.0712051, 0.0712055
5: -2.4017086, -1.8002263, -2.4017086, -1.8002263, -0.2723739, 0.2723505
6: -0.2971857, 0.1779544, -0.2971857, 0.1779544, -0.3864245, 0.3864413
7: -1.8941863, -1.2658451, -1.8941863, -1.2658451, -0.4500369, 0.4498346
8: -4.2648854, -3.2627838, -4.2648854, -3.2627838, -0.4637989, 0.4638130
9: -5.9969206, -4.8441453, -5.9969206, -4.8441453, -0.4464841, 0.4464754

Time for backsubstitution: 6.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 562
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 577
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 2854
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2859
type: DSZ, layer: 1, pos: 2874
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2901
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2900
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2902
type: DSZ, layer: 1, pos: 2845
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 2837
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2905
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 280
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 265
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 279
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 248
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 2821
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 310
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 3322
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 507
type: DSZ, layer: 1, pos: 3396
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 294
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 591
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 519
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2880
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2895
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2924
type: DSZ, layer: 1, pos: 3255
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3365
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3370
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3465
type: DSZ, layer: 1, pos: 3483

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3472

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0490751, upper bound: 0.0490875
time: 152.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0490751, upper bound: 0.0490879
time: 57.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.9899344, -1.6791523, -2.9899344, -1.6791523, -0.4808091, 0.4807868
1: -5.4404860, -3.4300203, -5.4404860, -3.4300203, -0.8269296, 0.8268890
2: -0.2036541, 0.1937240, -0.2036541, 0.1937240, -0.1201666, 0.1202161
3: -2.3926487, -1.7725695, -2.3926487, -1.7725695, -0.1879310, 0.1879257
4: 0.4635379, 0.5799251, 0.4635379, 0.5799251, -0.0712051, 0.0712055
5: -2.4017086, -1.8002263, -2.4017086, -1.8002263, -0.2723732, 0.2723512
6: -0.2971857, 0.1779544, -0.2971857, 0.1779544, -0.3864215, 0.3864443
7: -1.8941863, -1.2658451, -1.8941863, -1.2658451, -0.4500369, 0.4498346
8: -4.2648854, -3.2627838, -4.2648854, -3.2627838, -0.4637996, 0.4638123
9: -5.9969206, -4.8441453, -5.9969206, -4.8441453, -0.4464822, 0.4464773

Time for backsubstitution: 6.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 562
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 577
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 2854
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2859
type: DSZ, layer: 1, pos: 2874
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2901
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2900
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2902
type: DSZ, layer: 1, pos: 2845
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 2837
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2905
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 280
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 265
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 279
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 248
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 2821
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 310
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 3322
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 507
type: DSZ, layer: 1, pos: 3396
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 294
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 591
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 519
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2880
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2895
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2924
type: DSZ, layer: 1, pos: 3255
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3365
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3370
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3465
type: DSZ, layer: 1, pos: 3483

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3472

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0490751, upper bound: 0.0490884
time: 155.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0490751, upper bound: 0.0490883
time: 57.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.9899344, -1.6791523, -2.9899344, -1.6791523, -0.4808275, 0.4807686
1: -5.4404860, -3.4300203, -5.4404860, -3.4300203, -0.8269535, 0.8268652
2: -0.2036541, 0.1937240, -0.2036541, 0.1937240, -0.1201577, 0.1202265
3: -2.3926487, -1.7725695, -2.3926487, -1.7725695, -0.1879105, 0.1879463
4: 0.4635379, 0.5799251, 0.4635379, 0.5799251, -0.0712051, 0.0712055
5: -2.4017086, -1.8002263, -2.4017086, -1.8002263, -0.2723519, 0.2723731
6: -0.2971857, 0.1779544, -0.2971857, 0.1779544, -0.3864243, 0.3864416
7: -1.8941863, -1.2658451, -1.8941863, -1.2658451, -0.4499592, 0.4499249
8: -4.2648854, -3.2627838, -4.2648854, -3.2627838, -0.4638184, 0.4637942
9: -5.9969206, -4.8441453, -5.9969206, -4.8441453, -0.4464777, 0.4464817

Time for backsubstitution: 6.33 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 58.93 + 1746.82 = 1805.75 seconds
