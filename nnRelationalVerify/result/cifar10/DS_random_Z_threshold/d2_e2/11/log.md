## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 11)
Time budget: 3600 seconds
Split limit: 100
Threshold: 0.10251260279999999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3.9078836, -2.3747187, -3.9078836, -2.3747187, -1.3428798, 1.3428802)
1: (-1.6670785, 0.3596058, -1.6670785, 0.3596058, -1.8303232, 1.8303232)
2: (-0.6084201, -0.3018690, -0.6084201, -0.3018690, -0.2348176, 0.2348175)
3: (-0.3689309, 0.1840438, -0.3689309, 0.1840438, -0.4901261, 0.4901261)
4: (-2.4632726, -1.7337420, -2.4632726, -1.7337420, -0.2651898, 0.2651898)
5: (-0.6010512, 0.1139360, -0.6010512, 0.1139360, -0.6315912, 0.6315912)
6: (-3.3389983, -2.4709666, -3.3389983, -2.4709666, -0.2626626, 0.2626626)
7: (-0.3176667, 0.4359028, -0.3176667, 0.4359028, -0.6587952, 0.6587954)
8: (-4.6518116, -2.6710997, -4.6518116, -2.6710997, -1.0975242, 1.0975242)
9: (-1.8730564, 0.0038037, -1.8730564, 0.0038037, -1.4228096, 1.4228096)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.71 + 36.46 = 44.18 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.1029240, upper bound: 0.1029258

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 955
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 913
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 911
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 2871
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 623
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 926
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 970
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 988
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 941
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 939
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 971
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 919
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 914
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 1035
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 3455
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 940
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 969
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 954
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2909
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 968
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 956
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 953
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 924
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 938
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 322
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 925
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 261
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 2813
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 958
type: DSZ, layer: 1, pos: 986
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 584

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2971

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1029218, upper bound: 0.1029229
time: 219.47 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1029227, upper bound: 0.1029231
time: 24.88 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 244.37 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 244.37
Output dim: 3, lower bound: -0.1029218, upper bound: 0.1029229
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 244.37
Output dim: 3, lower bound: -0.1029227, upper bound: 0.1029231

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -3.9078836, -2.3747187, -3.9078836, -2.3747187, -1.3421664, 1.3421450
1: -1.6670785, 0.3596058, -1.6670785, 0.3596058, -1.8303132, 1.8303118
2: -0.6084201, -0.3018690, -0.6084201, -0.3018690, -0.2347894, 0.2347898
3: -0.3689309, 0.1840438, -0.3689309, 0.1840438, -0.4901234, 0.4901235
4: -2.4632726, -1.7337420, -2.4632726, -1.7337420, -0.2649945, 0.2649890
5: -0.6010512, 0.1139360, -0.6010512, 0.1139360, -0.6315744, 0.6315742
6: -3.3389983, -2.4709666, -3.3389983, -2.4709666, -0.2624133, 0.2624305
7: -0.3176667, 0.4359028, -0.3176667, 0.4359028, -0.6587944, 0.6587944
8: -4.6518116, -2.6710997, -4.6518116, -2.6710997, -1.0967987, 1.0967555
9: -1.8730564, 0.0038037, -1.8730564, 0.0038037, -1.4227815, 1.4227805

Time for backsubstitution: 5.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 970
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 261
type: DSZ, layer: 1, pos: 955
type: DSZ, layer: 1, pos: 919
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 913
type: DSZ, layer: 1, pos: 941
type: DSZ, layer: 1, pos: 2871
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 911
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 954
type: DSZ, layer: 1, pos: 2813
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2909
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 969
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 925
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 3455
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 924
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 940
type: DSZ, layer: 1, pos: 939
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 914
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 953
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 986
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 971
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 968
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 926
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 958
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 322
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 623
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 956
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 938
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 988
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 1035

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2287

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1029184, upper bound: 0.1029217
time: 297.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1029210, upper bound: 0.1029195
time: 194.34 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -3.9078836, -2.3747187, -3.9078836, -2.3747187, -1.3421459, 1.3421659
1: -1.6670785, 0.3596058, -1.6670785, 0.3596058, -1.8303123, 1.8303137
2: -0.6084201, -0.3018690, -0.6084201, -0.3018690, -0.2347898, 0.2347895
3: -0.3689309, 0.1840438, -0.3689309, 0.1840438, -0.4901235, 0.4901233
4: -2.4632726, -1.7337420, -2.4632726, -1.7337420, -0.2649890, 0.2649945
5: -0.6010512, 0.1139360, -0.6010512, 0.1139360, -0.6315742, 0.6315744
6: -3.3389983, -2.4709666, -3.3389983, -2.4709666, -0.2624305, 0.2624133
7: -0.3176667, 0.4359028, -0.3176667, 0.4359028, -0.6587945, 0.6587944
8: -4.6518116, -2.6710997, -4.6518116, -2.6710997, -1.0967560, 1.0967984
9: -1.8730564, 0.0038037, -1.8730564, 0.0038037, -1.4227805, 1.4227815

Time for backsubstitution: 5.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 2871
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 988
type: DSZ, layer: 1, pos: 911
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 3455
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 968
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 925
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2813
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 955
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 914
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 940
type: DSZ, layer: 1, pos: 261
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 1035
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 971
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 939
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 969
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 924
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 956
type: DSZ, layer: 1, pos: 2909
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 623
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 970
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 926
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 954
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 986
type: DSZ, layer: 1, pos: 941
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 322
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 938
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 958
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 919
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 953
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 913
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 2649

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2459

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1029246, upper bound: 0.1029205
time: 160.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1029246, upper bound: 0.1029267
time: 8.52 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 174.52 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 174.52
Output dim: 3, lower bound: -0.1029184, upper bound: 0.1029217
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 174.52
Output dim: 3, lower bound: -0.1029210, upper bound: 0.1029195
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 174.52
Output dim: 3, lower bound: -0.1029246, upper bound: 0.1029205
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 174.52
Output dim: 3, lower bound: -0.1029246, upper bound: 0.1029267

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.9078836, -2.3747187, -3.9078836, -2.3747187, -1.3422318, 1.3422060
1: -1.6670785, 0.3596058, -1.6670785, 0.3596058, -1.8302145, 1.8302093
2: -0.6084201, -0.3018690, -0.6084201, -0.3018690, -0.2347527, 0.2347736
3: -0.3689309, 0.1840438, -0.3689309, 0.1840438, -0.4901057, 0.4901069
4: -2.4632726, -1.7337420, -2.4632726, -1.7337420, -0.2649122, 0.2648954
5: -0.6010512, 0.1139360, -0.6010512, 0.1139360, -0.6315626, 0.6315639
6: -3.3389983, -2.4709666, -3.3389983, -2.4709666, -0.2622174, 0.2622677
7: -0.3176667, 0.4359028, -0.3176667, 0.4359028, -0.6588017, 0.6588006
8: -4.6518116, -2.6710997, -4.6518116, -2.6710997, -1.0966980, 1.0966415
9: -1.8730564, 0.0038037, -1.8730564, 0.0038037, -1.4223776, 1.4223781

Time for backsubstitution: 5.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 925
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 913
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 969
type: DSZ, layer: 1, pos: 971
type: DSZ, layer: 1, pos: 940
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 2909
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 986
type: DSZ, layer: 1, pos: 958
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 970
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 956
type: DSZ, layer: 1, pos: 926
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 988
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 3455
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 954
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 914
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 955
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 938
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2871
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 261
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 968
type: DSZ, layer: 1, pos: 941
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 919
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2813
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 1035
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 953
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 322
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 939
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 924
type: DSZ, layer: 1, pos: 911
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 623
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2349

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2462

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1029189, upper bound: 0.1029225
time: 131.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1029189, upper bound: 0.1029223
time: 133.04 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.9078836, -2.3747187, -3.9078836, -2.3747187, -1.3422270, 1.3422108
1: -1.6670785, 0.3596058, -1.6670785, 0.3596058, -1.8302107, 1.8302140
2: -0.6084201, -0.3018690, -0.6084201, -0.3018690, -0.2347732, 0.2347531
3: -0.3689309, 0.1840438, -0.3689309, 0.1840438, -0.4901069, 0.4901057
4: -2.4632726, -1.7337420, -2.4632726, -1.7337420, -0.2649009, 0.2649067
5: -0.6010512, 0.1139360, -0.6010512, 0.1139360, -0.6315639, 0.6315625
6: -3.3389983, -2.4709666, -3.3389983, -2.4709666, -0.2622505, 0.2622347
7: -0.3176667, 0.4359028, -0.3176667, 0.4359028, -0.6588006, 0.6588018
8: -4.6518116, -2.6710997, -4.6518116, -2.6710997, -1.0966842, 1.0966554
9: -1.8730564, 0.0038037, -1.8730564, 0.0038037, -1.4223785, 1.4223771

Time for backsubstitution: 6.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 956
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 939
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 924
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 954
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 261
type: DSZ, layer: 1, pos: 971
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 623
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 1035
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2871
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 322
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 940
type: DSZ, layer: 1, pos: 2909
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 986
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 968
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 953
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 926
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 941
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 925
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 970
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 3455
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 969
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 913
type: DSZ, layer: 1, pos: 919
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 914
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 988
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 938
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 2813
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 958
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 955
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 911
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2671

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 943

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1029213, upper bound: 0.1029194
time: 33.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1029215, upper bound: 0.1029184
time: 45.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.9078836, -2.3747187, -3.9078836, -2.3747187, -1.3421459, 1.3421659
1: -1.6670785, 0.3596058, -1.6670785, 0.3596058, -1.8303123, 1.8303137
2: -0.6084201, -0.3018690, -0.6084201, -0.3018690, -0.2347898, 0.2347895
3: -0.3689309, 0.1840438, -0.3689309, 0.1840438, -0.4901235, 0.4901233
4: -2.4632726, -1.7337420, -2.4632726, -1.7337420, -0.2649890, 0.2649945
5: -0.6010512, 0.1139360, -0.6010512, 0.1139360, -0.6315742, 0.6315744
6: -3.3389983, -2.4709666, -3.3389983, -2.4709666, -0.2624305, 0.2624133
7: -0.3176667, 0.4359028, -0.3176667, 0.4359028, -0.6587945, 0.6587944
8: -4.6518116, -2.6710997, -4.6518116, -2.6710997, -1.0967560, 1.0967984
9: -1.8730564, 0.0038037, -1.8730564, 0.0038037, -1.4227805, 1.4227815

Time for backsubstitution: 6.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 911
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 623
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 261
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 988
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 322
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 940
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 941
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 939
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 938
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 969
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 970
type: DSZ, layer: 1, pos: 986
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 971
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 955
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 919
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 914
type: DSZ, layer: 1, pos: 954
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 926
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 925
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 956
type: DSZ, layer: 1, pos: 2909
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 1035
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2871
type: DSZ, layer: 1, pos: 953
type: DSZ, layer: 1, pos: 913
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 958
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 968
type: DSZ, layer: 1, pos: 3455
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2813
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 924
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 3142

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 306

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1028820, upper bound: 0.1029218
time: 115.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1029237, upper bound: 0.1028809
time: 280.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.9078836, -2.3747187, -3.9078836, -2.3747187, -1.3421459, 1.3421659
1: -1.6670785, 0.3596058, -1.6670785, 0.3596058, -1.8303123, 1.8303137
2: -0.6084201, -0.3018690, -0.6084201, -0.3018690, -0.2347898, 0.2347895
3: -0.3689309, 0.1840438, -0.3689309, 0.1840438, -0.4901235, 0.4901233
4: -2.4632726, -1.7337420, -2.4632726, -1.7337420, -0.2649890, 0.2649945
5: -0.6010512, 0.1139360, -0.6010512, 0.1139360, -0.6315742, 0.6315744
6: -3.3389983, -2.4709666, -3.3389983, -2.4709666, -0.2624305, 0.2624133
7: -0.3176667, 0.4359028, -0.3176667, 0.4359028, -0.6587945, 0.6587944
8: -4.6518116, -2.6710997, -4.6518116, -2.6710997, -1.0967560, 1.0967984
9: -1.8730564, 0.0038037, -1.8730564, 0.0038037, -1.4227805, 1.4227815

Time for backsubstitution: 5.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2871
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 925
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 956
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 1035
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 322
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 986
type: DSZ, layer: 1, pos: 623
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 955
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2813
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 3455
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 913
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 914
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 924
type: DSZ, layer: 1, pos: 971
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 988
type: DSZ, layer: 1, pos: 968
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 970
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 958
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 938
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 926
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 939
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 911
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 941
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 953
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 261
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 954
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 919
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 940
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 969
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 2909
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2343

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 63

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1029077, upper bound: 0.1029030
time: 183.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1029041, upper bound: 0.1029094
time: 15.26 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 204.96 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 204.96
Output dim: 3, lower bound: -0.1029189, upper bound: 0.1029225
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 204.96
Output dim: 3, lower bound: -0.1029189, upper bound: 0.1029223
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 204.96
Output dim: 3, lower bound: -0.1029213, upper bound: 0.1029194
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 204.96
Output dim: 3, lower bound: -0.1029215, upper bound: 0.1029184
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 204.96
Output dim: 3, lower bound: -0.1028820, upper bound: 0.1029218
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 204.96
Output dim: 3, lower bound: -0.1029237, upper bound: 0.1028809
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 204.96
Output dim: 3, lower bound: -0.1029077, upper bound: 0.1029030
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 204.96
Output dim: 3, lower bound: -0.1029041, upper bound: 0.1029094

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.9078836, -2.3747187, -3.9078836, -2.3747187, -1.3422318, 1.3422060
1: -1.6670785, 0.3596058, -1.6670785, 0.3596058, -1.8302145, 1.8302093
2: -0.6084201, -0.3018690, -0.6084201, -0.3018690, -0.2347527, 0.2347736
3: -0.3689309, 0.1840438, -0.3689309, 0.1840438, -0.4901057, 0.4901069
4: -2.4632726, -1.7337420, -2.4632726, -1.7337420, -0.2649122, 0.2648954
5: -0.6010512, 0.1139360, -0.6010512, 0.1139360, -0.6315626, 0.6315639
6: -3.3389983, -2.4709666, -3.3389983, -2.4709666, -0.2622174, 0.2622677
7: -0.3176667, 0.4359028, -0.3176667, 0.4359028, -0.6588017, 0.6588006
8: -4.6518116, -2.6710997, -4.6518116, -2.6710997, -1.0966980, 1.0966415
9: -1.8730564, 0.0038037, -1.8730564, 0.0038037, -1.4223776, 1.4223781

Time for backsubstitution: 6.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 3455
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 1035
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 955
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2871
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 2909
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 926
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 988
type: DSZ, layer: 1, pos: 971
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 924
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 953
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 925
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 919
type: DSZ, layer: 1, pos: 969
type: DSZ, layer: 1, pos: 261
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 322
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 911
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 958
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 623
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 956
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 954
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 940
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 970
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 938
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 968
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 914
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 913
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 986
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 941
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 939
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2813
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 957

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 742

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1028982, upper bound: 0.1029203
time: 130.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1029173, upper bound: 0.1029034
time: 164.92 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.9078836, -2.3747187, -3.9078836, -2.3747187, -1.3422318, 1.3422060
1: -1.6670785, 0.3596058, -1.6670785, 0.3596058, -1.8302145, 1.8302093
2: -0.6084201, -0.3018690, -0.6084201, -0.3018690, -0.2347527, 0.2347736
3: -0.3689309, 0.1840438, -0.3689309, 0.1840438, -0.4901057, 0.4901069
4: -2.4632726, -1.7337420, -2.4632726, -1.7337420, -0.2649122, 0.2648954
5: -0.6010512, 0.1139360, -0.6010512, 0.1139360, -0.6315626, 0.6315639
6: -3.3389983, -2.4709666, -3.3389983, -2.4709666, -0.2622174, 0.2622677
7: -0.3176667, 0.4359028, -0.3176667, 0.4359028, -0.6588017, 0.6588006
8: -4.6518116, -2.6710997, -4.6518116, -2.6710997, -1.0966980, 1.0966415
9: -1.8730564, 0.0038037, -1.8730564, 0.0038037, -1.4223776, 1.4223781

Time for backsubstitution: 6.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 968
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 954
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 953
type: DSZ, layer: 1, pos: 914
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 925
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 988
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 939
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2813
type: DSZ, layer: 1, pos: 955
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 940
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2909
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 971
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 941
type: DSZ, layer: 1, pos: 623
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 913
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 938
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 958
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 911
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 1035
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 924
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 969
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 926
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 261
type: DSZ, layer: 1, pos: 2871
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 919
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 986
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 970
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 3455
type: DSZ, layer: 1, pos: 956
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 322
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 890

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3494

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1029265, upper bound: 0.1029226
time: 153.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1029265, upper bound: 0.1029232
time: 283.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.9078836, -2.3747187, -3.9078836, -2.3747187, -1.3422270, 1.3422108
1: -1.6670785, 0.3596058, -1.6670785, 0.3596058, -1.8302102, 1.8302135
2: -0.6084201, -0.3018690, -0.6084201, -0.3018690, -0.2347732, 0.2347530
3: -0.3689309, 0.1840438, -0.3689309, 0.1840438, -0.4901067, 0.4901055
4: -2.4632726, -1.7337420, -2.4632726, -1.7337420, -0.2649009, 0.2649067
5: -0.6010512, 0.1139360, -0.6010512, 0.1139360, -0.6315639, 0.6315625
6: -3.3389983, -2.4709666, -3.3389983, -2.4709666, -0.2622505, 0.2622347
7: -0.3176667, 0.4359028, -0.3176667, 0.4359028, -0.6588006, 0.6588018
8: -4.6518116, -2.6710997, -4.6518116, -2.6710997, -1.0966842, 1.0966554
9: -1.8730564, 0.0038037, -1.8730564, 0.0038037, -1.4223785, 1.4223776

Time for backsubstitution: 6.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 911
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 2813
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 955
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 988
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 958
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 939
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 3455
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 914
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 969
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 941
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 971
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 1035
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 953
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 924
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 2909
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 919
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 986
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 261
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 954
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 322
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 925
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 940
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 956
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 913
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2871
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 623
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 938
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 968
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 970
type: DSZ, layer: 1, pos: 926
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 910

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2470

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1029212, upper bound: 0.1029219
time: 16.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1029212, upper bound: 0.1029172
time: 200.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.9078836, -2.3747187, -3.9078836, -2.3747187, -1.3422270, 1.3422108
1: -1.6670785, 0.3596058, -1.6670785, 0.3596058, -1.8302102, 1.8302135
2: -0.6084201, -0.3018690, -0.6084201, -0.3018690, -0.2347732, 0.2347530
3: -0.3689309, 0.1840438, -0.3689309, 0.1840438, -0.4901067, 0.4901055
4: -2.4632726, -1.7337420, -2.4632726, -1.7337420, -0.2649009, 0.2649067
5: -0.6010512, 0.1139360, -0.6010512, 0.1139360, -0.6315639, 0.6315625
6: -3.3389983, -2.4709666, -3.3389983, -2.4709666, -0.2622505, 0.2622347
7: -0.3176667, 0.4359028, -0.3176667, 0.4359028, -0.6588006, 0.6588018
8: -4.6518116, -2.6710997, -4.6518116, -2.6710997, -1.0966842, 1.0966554
9: -1.8730564, 0.0038037, -1.8730564, 0.0038037, -1.4223785, 1.4223776

Time for backsubstitution: 6.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 3455
type: DSZ, layer: 1, pos: 924
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 954
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 261
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 986
type: DSZ, layer: 1, pos: 938
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 914
type: DSZ, layer: 1, pos: 939
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 623
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 926
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 970
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2909
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 958
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 2871
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 919
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 988
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 322
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 940
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 1035
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 955
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 956
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 971
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 306
type: DSZ, layer: 1, pos: 2813
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 913
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 941
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 911
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 969
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 953
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 968
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 925
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 928

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 800

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1028874, upper bound: 0.1028844
time: 286.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1028859, upper bound: 0.1028850
time: 267.86 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.9078836, -2.3747187, -3.9078836, -2.3747187, -1.3421392, 1.3421597
1: -1.6670785, 0.3596058, -1.6670785, 0.3596058, -1.8302307, 1.8302345
2: -0.6084201, -0.3018690, -0.6084201, -0.3018690, -0.2347658, 0.2347689
3: -0.3689309, 0.1840438, -0.3689309, 0.1840438, -0.4899881, 0.4900030
4: -2.4632726, -1.7337420, -2.4632726, -1.7337420, -0.2637780, 0.2639314
5: -0.6010512, 0.1139360, -0.6010512, 0.1139360, -0.6313330, 0.6313686
6: -3.3389983, -2.4709666, -3.3389983, -2.4709666, -0.2614331, 0.2615810
7: -0.3176667, 0.4359028, -0.3176667, 0.4359028, -0.6583910, 0.6583586
8: -4.6518116, -2.6710997, -4.6518116, -2.6710997, -1.0966434, 1.0966794
9: -1.8730564, 0.0038037, -1.8730564, 0.0038037, -1.4227777, 1.4227781

Time for backsubstitution: 6.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2344
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 957
type: DSZ, layer: 1, pos: 955
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 889
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2871
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2884
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 956
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2844
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 640
type: DSZ, layer: 1, pos: 2970
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 924
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 3073
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 515
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 1035
type: DSZ, layer: 1, pos: 2287
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 322
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3455
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 261
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 913
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 958
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2887
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 2301
type: DSZ, layer: 1, pos: 954
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 925
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 988
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 953
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2340
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 2881
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 968
type: DSZ, layer: 1, pos: 971
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 970
type: DSZ, layer: 1, pos: 939
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 941
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 926
type: DSZ, layer: 1, pos: 2689
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 63
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 938
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 614
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2396
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2885
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 623
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 914
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 2883
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 911
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 940
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2813
type: DSZ, layer: 1, pos: 271
type: DSZ, layer: 1, pos: 986
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 2909
type: DSZ, layer: 1, pos: 919
type: DSZ, layer: 1, pos: 969
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 3031
type: DSZ, layer: 1, pos: 812
type: DSZ, layer: 1, pos: 2481
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 2355

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2344

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1027106, upper bound: 0.1029109
time: 148.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1028736, upper bound: 0.1027506
time: 212.69 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 367.44 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 367.44
Output dim: 3, lower bound: -0.1028982, upper bound: 0.1029203
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 367.44
Output dim: 3, lower bound: -0.1029173, upper bound: 0.1029034
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 367.44
Output dim: 3, lower bound: -0.1029265, upper bound: 0.1029226
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 367.44
Output dim: 3, lower bound: -0.1029265, upper bound: 0.1029232
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 367.44
Output dim: 3, lower bound: -0.1029212, upper bound: 0.1029219
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 367.44
Output dim: 3, lower bound: -0.1029212, upper bound: 0.1029172
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 367.44
Output dim: 3, lower bound: -0.1028874, upper bound: 0.1028844
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 367.44
Output dim: 3, lower bound: -0.1028859, upper bound: 0.1028850
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 367.44
Output dim: 3, lower bound: -0.1027106, upper bound: 0.1029109
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 367.44
Output dim: 3, lower bound: -0.1028736, upper bound: 0.1027506
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 367.44
Output dim: 3, lower bound: -0.1029237, upper bound: 0.1028809
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 367.44
Output dim: 3, lower bound: -0.1029077, upper bound: 0.1029030
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 367.44
Output dim: 3, lower bound: -0.1029041, upper bound: 0.1029094

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 44.18 + 3774.61 = 3818.78 seconds
