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
execution time: IAR + RelationalAnalysis = 8.19 + 52.45 = 60.64 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.0491315, upper bound: 0.0491336

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3370
type: DSZ, layer: 1, pos: 280
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 294
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2900
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2821
type: DSZ, layer: 1, pos: 2924
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 3365
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 591
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 507
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 265
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 2854
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2905
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 2880
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2874
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 3396
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2837
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2895
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 310
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2901
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 2859
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3255
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 248
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 577
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 279
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 3465
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 519
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 3322
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 562
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2902
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2845
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 3362

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3370

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0491327, upper bound: 0.0491346
time: 64.45 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0491304, upper bound: 0.0491332
time: 244.68 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 309.15 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 309.15
Output dim: 4, lower bound: -0.0491327, upper bound: 0.0491346
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 309.15
Output dim: 4, lower bound: -0.0491304, upper bound: 0.0491332

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -2.9899344, -1.6791523, -2.9899344, -1.6791523, -0.4830543, 0.4830543
1: -5.4404860, -3.4300203, -5.4404860, -3.4300203, -0.8298308, 0.8298308
2: -0.2036541, 0.1937240, -0.2036541, 0.1937240, -0.1220713, 0.1220713
3: -2.3926487, -1.7725695, -2.3926487, -1.7725695, -0.1881570, 0.1881570
4: 0.4635379, 0.5799251, 0.4635379, 0.5799251, -0.0711872, 0.0711872
5: -2.4017086, -1.8002263, -2.4017086, -1.8002263, -0.2739069, 0.2739069
6: -0.2971857, 0.1779544, -0.2971857, 0.1779544, -0.3865951, 0.3865951
7: -1.8941863, -1.2658451, -1.8941863, -1.2658451, -0.4546693, 0.4546692
8: -4.2648854, -3.2627838, -4.2648854, -3.2627838, -0.4648452, 0.4648451
9: -5.9969206, -4.8441453, -5.9969206, -4.8441453, -0.4469083, 0.4469082

Time for backsubstitution: 6.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 3322
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2845
type: DSZ, layer: 1, pos: 591
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2924
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 3365
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 562
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 248
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2900
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2854
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 2905
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2874
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 265
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2895
type: DSZ, layer: 1, pos: 3465
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 507
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 3255
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2821
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2837
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 3396
type: DSZ, layer: 1, pos: 2880
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 280
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 519
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 294
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 310
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 279
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 577
type: DSZ, layer: 1, pos: 2859
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2902
type: DSZ, layer: 1, pos: 2901
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 3459

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2086

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0491264, upper bound: 0.0491294
time: 201.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0491268, upper bound: 0.0491300
time: 68.58 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -2.9899344, -1.6791523, -2.9899344, -1.6791523, -0.4830543, 0.4830543
1: -5.4404860, -3.4300203, -5.4404860, -3.4300203, -0.8298308, 0.8298308
2: -0.2036541, 0.1937240, -0.2036541, 0.1937240, -0.1220713, 0.1220713
3: -2.3926487, -1.7725695, -2.3926487, -1.7725695, -0.1881570, 0.1881570
4: 0.4635379, 0.5799251, 0.4635379, 0.5799251, -0.0711872, 0.0711872
5: -2.4017086, -1.8002263, -2.4017086, -1.8002263, -0.2739069, 0.2739069
6: -0.2971857, 0.1779544, -0.2971857, 0.1779544, -0.3865951, 0.3865951
7: -1.8941863, -1.2658451, -1.8941863, -1.2658451, -0.4546693, 0.4546692
8: -4.2648854, -3.2627838, -4.2648854, -3.2627838, -0.4648452, 0.4648451
9: -5.9969206, -4.8441453, -5.9969206, -4.8441453, -0.4469083, 0.4469082

Time for backsubstitution: 6.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3365
type: DSZ, layer: 1, pos: 310
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2880
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 591
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 280
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 2905
type: DSZ, layer: 1, pos: 2924
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 3322
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 519
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 3465
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2845
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2821
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 2900
type: DSZ, layer: 1, pos: 279
type: DSZ, layer: 1, pos: 248
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 294
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2837
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2902
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 562
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2874
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 577
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 3255
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 3396
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2859
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2901
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 2895
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 507
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 265
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 2854

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3365

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0491321, upper bound: 0.0491345
time: 133.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0491307, upper bound: 0.0491335
time: 150.88 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 290.35 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 290.35
Output dim: 4, lower bound: -0.0491264, upper bound: 0.0491294
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 290.35
Output dim: 4, lower bound: -0.0491268, upper bound: 0.0491300
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 290.35
Output dim: 4, lower bound: -0.0491321, upper bound: 0.0491345
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 290.35
Output dim: 4, lower bound: -0.0491307, upper bound: 0.0491335

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.9899344, -1.6791523, -2.9899344, -1.6791523, -0.4823698, 0.4823383
1: -5.4404860, -3.4300203, -5.4404860, -3.4300203, -0.8296398, 0.8296356
2: -0.2036541, 0.1937240, -0.2036541, 0.1937240, -0.1217413, 0.1217499
3: -2.3926487, -1.7725695, -2.3926487, -1.7725695, -0.1881149, 0.1881163
4: 0.4635379, 0.5799251, 0.4635379, 0.5799251, -0.0711854, 0.0711857
5: -2.4017086, -1.8002263, -2.4017086, -1.8002263, -0.2738716, 0.2738732
6: -0.2971857, 0.1779544, -0.2971857, 0.1779544, -0.3865950, 0.3865950
7: -1.8941863, -1.2658451, -1.8941863, -1.2658451, -0.4546310, 0.4546301
8: -4.2648854, -3.2627838, -4.2648854, -3.2627838, -0.4634043, 0.4633290
9: -5.9969206, -4.8441453, -5.9969206, -4.8441453, -0.4464416, 0.4464548

Time for backsubstitution: 6.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 591
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2901
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2905
type: DSZ, layer: 1, pos: 3322
type: DSZ, layer: 1, pos: 294
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 248
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 3255
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2845
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 2924
type: DSZ, layer: 1, pos: 280
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 2902
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 3365
type: DSZ, layer: 1, pos: 265
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2854
type: DSZ, layer: 1, pos: 577
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2874
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 310
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 2900
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2895
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2880
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 279
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2859
type: DSZ, layer: 1, pos: 3396
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 3465
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 2837
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 519
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 507
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 562
type: DSZ, layer: 1, pos: 2821

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0491253, upper bound: 0.0491289
time: 134.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0491253, upper bound: 0.0491303
time: 14.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.9899344, -1.6791523, -2.9899344, -1.6791523, -0.4823384, 0.4823698
1: -5.4404860, -3.4300203, -5.4404860, -3.4300203, -0.8296356, 0.8296398
2: -0.2036541, 0.1937240, -0.2036541, 0.1937240, -0.1217499, 0.1217413
3: -2.3926487, -1.7725695, -2.3926487, -1.7725695, -0.1881163, 0.1881149
4: 0.4635379, 0.5799251, 0.4635379, 0.5799251, -0.0711857, 0.0711854
5: -2.4017086, -1.8002263, -2.4017086, -1.8002263, -0.2738732, 0.2738717
6: -0.2971857, 0.1779544, -0.2971857, 0.1779544, -0.3865951, 0.3865950
7: -1.8941863, -1.2658451, -1.8941863, -1.2658451, -0.4546301, 0.4546311
8: -4.2648854, -3.2627838, -4.2648854, -3.2627838, -0.4633289, 0.4634042
9: -5.9969206, -4.8441453, -5.9969206, -4.8441453, -0.4464548, 0.4464416

Time for backsubstitution: 6.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 519
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 294
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 310
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2924
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2905
type: DSZ, layer: 1, pos: 265
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 3322
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 279
type: DSZ, layer: 1, pos: 2902
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2845
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2874
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2900
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 280
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 3365
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 577
type: DSZ, layer: 1, pos: 2901
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2854
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2880
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 562
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 3255
type: DSZ, layer: 1, pos: 2859
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 2837
type: DSZ, layer: 1, pos: 3465
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2821
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 591
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 3396
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 248
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 507
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2895
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2059

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 263

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0490036, upper bound: 0.0491144
time: 134.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0491099, upper bound: 0.0490087
time: 19.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.9899344, -1.6791523, -2.9899344, -1.6791523, -0.4830543, 0.4830543
1: -5.4404860, -3.4300203, -5.4404860, -3.4300203, -0.8298308, 0.8298308
2: -0.2036541, 0.1937240, -0.2036541, 0.1937240, -0.1220713, 0.1220713
3: -2.3926487, -1.7725695, -2.3926487, -1.7725695, -0.1881570, 0.1881570
4: 0.4635379, 0.5799251, 0.4635379, 0.5799251, -0.0711872, 0.0711872
5: -2.4017086, -1.8002263, -2.4017086, -1.8002263, -0.2739069, 0.2739069
6: -0.2971857, 0.1779544, -0.2971857, 0.1779544, -0.3865951, 0.3865951
7: -1.8941863, -1.2658451, -1.8941863, -1.2658451, -0.4546693, 0.4546692
8: -4.2648854, -3.2627838, -4.2648854, -3.2627838, -0.4648452, 0.4648451
9: -5.9969206, -4.8441453, -5.9969206, -4.8441453, -0.4469083, 0.4469082

Time for backsubstitution: 6.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 248
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 2901
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 265
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2874
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 280
type: DSZ, layer: 1, pos: 2902
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 591
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2854
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2900
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 3322
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2895
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2837
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2905
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2859
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 2821
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 279
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 310
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 294
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 507
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2924
type: DSZ, layer: 1, pos: 2845
type: DSZ, layer: 1, pos: 3255
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 3465
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2880
type: DSZ, layer: 1, pos: 3396
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 562
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 519
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 577

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2030

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0491306, upper bound: 0.0491336
time: 298.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0491294, upper bound: 0.0491320
time: 41.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.9899344, -1.6791523, -2.9899344, -1.6791523, -0.4830543, 0.4830543
1: -5.4404860, -3.4300203, -5.4404860, -3.4300203, -0.8298308, 0.8298308
2: -0.2036541, 0.1937240, -0.2036541, 0.1937240, -0.1220713, 0.1220713
3: -2.3926487, -1.7725695, -2.3926487, -1.7725695, -0.1881570, 0.1881570
4: 0.4635379, 0.5799251, 0.4635379, 0.5799251, -0.0711872, 0.0711872
5: -2.4017086, -1.8002263, -2.4017086, -1.8002263, -0.2739069, 0.2739069
6: -0.2971857, 0.1779544, -0.2971857, 0.1779544, -0.3865951, 0.3865951
7: -1.8941863, -1.2658451, -1.8941863, -1.2658451, -0.4546693, 0.4546692
8: -4.2648854, -3.2627838, -4.2648854, -3.2627838, -0.4648452, 0.4648451
9: -5.9969206, -4.8441453, -5.9969206, -4.8441453, -0.4469083, 0.4469082

Time for backsubstitution: 6.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 3322
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 577
type: DSZ, layer: 1, pos: 2851
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2845
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2840
type: DSZ, layer: 1, pos: 248
type: DSZ, layer: 1, pos: 562
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 294
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 605
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 2880
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2837
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2854
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 3349
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3255
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 265
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 3396
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 2821
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 546
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 591
type: DSZ, layer: 1, pos: 2841
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 507
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2924
type: DSZ, layer: 1, pos: 3257
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 279
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 263
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 3369
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 2895
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 2905
type: DSZ, layer: 1, pos: 2874
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3465
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2826
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2859
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 280
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 262
type: DSZ, layer: 1, pos: 2900
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2839
type: DSZ, layer: 1, pos: 519
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 310
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 3372
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2902
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 354
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 2901
type: DSZ, layer: 1, pos: 2070

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2142

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0491149, upper bound: 0.0491168
time: 174.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0491169, upper bound: 0.0491181
time: 102.65 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 283.56 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 283.56
Output dim: 4, lower bound: -0.0491253, upper bound: 0.0491289
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 283.56
Output dim: 4, lower bound: -0.0491253, upper bound: 0.0491303
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 283.56
Output dim: 4, lower bound: -0.0490036, upper bound: 0.0491144
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 283.56
Output dim: 4, lower bound: -0.0491099, upper bound: 0.0490087
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 283.56
Output dim: 4, lower bound: -0.0491306, upper bound: 0.0491336
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 283.56
Output dim: 4, lower bound: -0.0491294, upper bound: 0.0491320
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 283.56
Output dim: 4, lower bound: -0.0491149, upper bound: 0.0491168
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 283.56
Output dim: 4, lower bound: -0.0491169, upper bound: 0.0491181

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 60.64 + 1820.08 = 1880.72 seconds
