## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 14)
Time budget: 3600 seconds
Split limit: 100
Threshold: 0.0399961638


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3.8308637, -2.7324901, -3.8308637, -2.7324901, -0.6773981, 0.6773980)
1: (-4.8651872, -3.0135963, -4.8651872, -3.0135963, -0.9633368, 0.9633367)
2: (-0.7754370, -0.5610399, -0.7754370, -0.5610399, -0.1354368, 0.1354368)
3: (0.1295116, 0.4294550, 0.1295116, 0.4294550, -0.1016447, 0.1016447)
4: (-1.0549200, -0.4258140, -1.0549200, -0.4258140, -0.2309317, 0.2309317)
5: (0.2908296, 0.5544203, 0.2908296, 0.5544203, -0.1145877, 0.1145877)
6: (-2.8310716, -1.9714756, -2.8310716, -1.9714756, -0.2252266, 0.2252266)
7: (0.7645508, 1.2200117, 0.7645508, 1.2200117, -0.0829536, 0.0829536)
8: (-4.1439810, -2.5720947, -4.1439810, -2.5720947, -0.6319618, 0.6319618)
9: (-3.3517365, -1.9905157, -3.3517365, -1.9905157, -0.6366515, 0.6366515)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.38 + 29.31 = 36.69 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0400272, upper bound: 0.0400339

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 439
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 675

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3142

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400270, upper bound: 0.0400291
time: 116.92 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400270, upper bound: 0.0400355
time: 13.42 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 130.35 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 130.35
Output dim: 7, lower bound: -0.0400270, upper bound: 0.0400291
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 130.35
Output dim: 7, lower bound: -0.0400270, upper bound: 0.0400355

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -3.8308637, -2.7324901, -3.8308637, -2.7324901, -0.6773981, 0.6773980
1: -4.8651872, -3.0135963, -4.8651872, -3.0135963, -0.9633368, 0.9633367
2: -0.7754370, -0.5610399, -0.7754370, -0.5610399, -0.1354368, 0.1354368
3: 0.1295116, 0.4294550, 0.1295116, 0.4294550, -0.1016447, 0.1016447
4: -1.0549200, -0.4258140, -1.0549200, -0.4258140, -0.2309317, 0.2309317
5: 0.2908296, 0.5544203, 0.2908296, 0.5544203, -0.1145877, 0.1145877
6: -2.8310716, -1.9714756, -2.8310716, -1.9714756, -0.2252266, 0.2252266
7: 0.7645508, 1.2200117, 0.7645508, 1.2200117, -0.0829536, 0.0829536
8: -4.1439810, -2.5720947, -4.1439810, -2.5720947, -0.6319618, 0.6319618
9: -3.3517365, -1.9905157, -3.3517365, -1.9905157, -0.6366515, 0.6366515

Time for backsubstitution: 5.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 439
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 3025

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2114

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400252, upper bound: 0.0400306
time: 116.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400252, upper bound: 0.0400285
time: 63.49 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -3.8308637, -2.7324901, -3.8308637, -2.7324901, -0.6773981, 0.6773980
1: -4.8651872, -3.0135963, -4.8651872, -3.0135963, -0.9633368, 0.9633367
2: -0.7754370, -0.5610399, -0.7754370, -0.5610399, -0.1354368, 0.1354368
3: 0.1295116, 0.4294550, 0.1295116, 0.4294550, -0.1016447, 0.1016447
4: -1.0549200, -0.4258140, -1.0549200, -0.4258140, -0.2309317, 0.2309317
5: 0.2908296, 0.5544203, 0.2908296, 0.5544203, -0.1145877, 0.1145877
6: -2.8310716, -1.9714756, -2.8310716, -1.9714756, -0.2252266, 0.2252266
7: 0.7645508, 1.2200117, 0.7645508, 1.2200117, -0.0829536, 0.0829536
8: -4.1439810, -2.5720947, -4.1439810, -2.5720947, -0.6319618, 0.6319618
9: -3.3517365, -1.9905157, -3.3517365, -1.9905157, -0.6366515, 0.6366515

Time for backsubstitution: 5.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 439
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2955

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2288

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400251, upper bound: 0.0400269
time: 79.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400239, upper bound: 0.0400284
time: 146.29 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 230.93 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 230.93
Output dim: 7, lower bound: -0.0400252, upper bound: 0.0400306
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 230.93
Output dim: 7, lower bound: -0.0400252, upper bound: 0.0400285
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 230.93
Output dim: 7, lower bound: -0.0400251, upper bound: 0.0400269
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 230.93
Output dim: 7, lower bound: -0.0400239, upper bound: 0.0400284

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.8308637, -2.7324901, -3.8308637, -2.7324901, -0.6773981, 0.6773980
1: -4.8651872, -3.0135963, -4.8651872, -3.0135963, -0.9633368, 0.9633367
2: -0.7754370, -0.5610399, -0.7754370, -0.5610399, -0.1354368, 0.1354368
3: 0.1295116, 0.4294550, 0.1295116, 0.4294550, -0.1016447, 0.1016447
4: -1.0549200, -0.4258140, -1.0549200, -0.4258140, -0.2309317, 0.2309317
5: 0.2908296, 0.5544203, 0.2908296, 0.5544203, -0.1145877, 0.1145877
6: -2.8310716, -1.9714756, -2.8310716, -1.9714756, -0.2252266, 0.2252266
7: 0.7645508, 1.2200117, 0.7645508, 1.2200117, -0.0829536, 0.0829536
8: -4.1439810, -2.5720947, -4.1439810, -2.5720947, -0.6319618, 0.6319618
9: -3.3517365, -1.9905157, -3.3517365, -1.9905157, -0.6366515, 0.6366515

Time for backsubstitution: 5.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 439
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 441

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 766

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400243, upper bound: 0.0400320
time: 53.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400256, upper bound: 0.0400242
time: 71.26 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.8308637, -2.7324901, -3.8308637, -2.7324901, -0.6773981, 0.6773980
1: -4.8651872, -3.0135963, -4.8651872, -3.0135963, -0.9633368, 0.9633367
2: -0.7754370, -0.5610399, -0.7754370, -0.5610399, -0.1354368, 0.1354368
3: 0.1295116, 0.4294550, 0.1295116, 0.4294550, -0.1016447, 0.1016447
4: -1.0549200, -0.4258140, -1.0549200, -0.4258140, -0.2309317, 0.2309317
5: 0.2908296, 0.5544203, 0.2908296, 0.5544203, -0.1145877, 0.1145877
6: -2.8310716, -1.9714756, -2.8310716, -1.9714756, -0.2252266, 0.2252266
7: 0.7645508, 1.2200117, 0.7645508, 1.2200117, -0.0829536, 0.0829536
8: -4.1439810, -2.5720947, -4.1439810, -2.5720947, -0.6319618, 0.6319618
9: -3.3517365, -1.9905157, -3.3517365, -1.9905157, -0.6366515, 0.6366515

Time for backsubstitution: 5.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 439
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2986

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2937

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400256, upper bound: 0.0400315
time: 17.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400255, upper bound: 0.0400265
time: 124.44 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.8308637, -2.7324901, -3.8308637, -2.7324901, -0.6773313, 0.6773241
1: -4.8651872, -3.0135963, -4.8651872, -3.0135963, -0.9627874, 0.9627538
2: -0.7754370, -0.5610399, -0.7754370, -0.5610399, -0.1354286, 0.1354287
3: 0.1295116, 0.4294550, 0.1295116, 0.4294550, -0.1016413, 0.1016412
4: -1.0549200, -0.4258140, -1.0549200, -0.4258140, -0.2306667, 0.2306770
5: 0.2908296, 0.5544203, 0.2908296, 0.5544203, -0.1145848, 0.1145847
6: -2.8310716, -1.9714756, -2.8310716, -1.9714756, -0.2251357, 0.2251378
7: 0.7645508, 1.2200117, 0.7645508, 1.2200117, -0.0829534, 0.0829534
8: -4.1439810, -2.5720947, -4.1439810, -2.5720947, -0.6318565, 0.6318593
9: -3.3517365, -1.9905157, -3.3517365, -1.9905157, -0.6360731, 0.6360239

Time for backsubstitution: 5.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 439
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 424

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3055

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400179, upper bound: 0.0400215
time: 18.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400153, upper bound: 0.0400216
time: 20.01 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.8308637, -2.7324901, -3.8308637, -2.7324901, -0.6773241, 0.6773312
1: -4.8651872, -3.0135963, -4.8651872, -3.0135963, -0.9627538, 0.9627874
2: -0.7754370, -0.5610399, -0.7754370, -0.5610399, -0.1354287, 0.1354285
3: 0.1295116, 0.4294550, 0.1295116, 0.4294550, -0.1016412, 0.1016413
4: -1.0549200, -0.4258140, -1.0549200, -0.4258140, -0.2306770, 0.2306667
5: 0.2908296, 0.5544203, 0.2908296, 0.5544203, -0.1145847, 0.1145848
6: -2.8310716, -1.9714756, -2.8310716, -1.9714756, -0.2251378, 0.2251357
7: 0.7645508, 1.2200117, 0.7645508, 1.2200117, -0.0829534, 0.0829534
8: -4.1439810, -2.5720947, -4.1439810, -2.5720947, -0.6318593, 0.6318564
9: -3.3517365, -1.9905157, -3.3517365, -1.9905157, -0.6360239, 0.6360732

Time for backsubstitution: 5.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 439
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2962

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400239, upper bound: 0.0400275
time: 17.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400208, upper bound: 0.0400253
time: 45.12 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 68.47 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 68.47
Output dim: 7, lower bound: -0.0400243, upper bound: 0.0400320
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 68.47
Output dim: 7, lower bound: -0.0400256, upper bound: 0.0400242
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 68.47
Output dim: 7, lower bound: -0.0400256, upper bound: 0.0400315
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 68.47
Output dim: 7, lower bound: -0.0400255, upper bound: 0.0400265
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 68.47
Output dim: 7, lower bound: -0.0400179, upper bound: 0.0400215
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 68.47
Output dim: 7, lower bound: -0.0400153, upper bound: 0.0400216
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 68.47
Output dim: 7, lower bound: -0.0400239, upper bound: 0.0400275
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 68.47
Output dim: 7, lower bound: -0.0400208, upper bound: 0.0400253

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.8308637, -2.7324901, -3.8308637, -2.7324901, -0.6773974, 0.6773603
1: -4.8651872, -3.0135963, -4.8651872, -3.0135963, -0.9633337, 0.9633114
2: -0.7754370, -0.5610399, -0.7754370, -0.5610399, -0.1354367, 0.1354357
3: 0.1295116, 0.4294550, 0.1295116, 0.4294550, -0.1016446, 0.1016446
4: -1.0549200, -0.4258140, -1.0549200, -0.4258140, -0.2309331, 0.2309277
5: 0.2908296, 0.5544203, 0.2908296, 0.5544203, -0.1145869, 0.1145879
6: -2.8310716, -1.9714756, -2.8310716, -1.9714756, -0.2252122, 0.2252219
7: 0.7645508, 1.2200117, 0.7645508, 1.2200117, -0.0829523, 0.0829528
8: -4.1439810, -2.5720947, -4.1439810, -2.5720947, -0.6319611, 0.6318690
9: -3.3517365, -1.9905157, -3.3517365, -1.9905157, -0.6366467, 0.6365998

Time for backsubstitution: 5.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 439
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 2336

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2151

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400245, upper bound: 0.0400164
time: 90.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400099, upper bound: 0.0400280
time: 18.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.8308637, -2.7324901, -3.8308637, -2.7324901, -0.6773604, 0.6773975
1: -4.8651872, -3.0135963, -4.8651872, -3.0135963, -0.9633116, 0.9633336
2: -0.7754370, -0.5610399, -0.7754370, -0.5610399, -0.1354357, 0.1354367
3: 0.1295116, 0.4294550, 0.1295116, 0.4294550, -0.1016446, 0.1016446
4: -1.0549200, -0.4258140, -1.0549200, -0.4258140, -0.2309277, 0.2309331
5: 0.2908296, 0.5544203, 0.2908296, 0.5544203, -0.1145879, 0.1145869
6: -2.8310716, -1.9714756, -2.8310716, -1.9714756, -0.2252218, 0.2252121
7: 0.7645508, 1.2200117, 0.7645508, 1.2200117, -0.0829528, 0.0829523
8: -4.1439810, -2.5720947, -4.1439810, -2.5720947, -0.6318691, 0.6319610
9: -3.3517365, -1.9905157, -3.3517365, -1.9905157, -0.6365998, 0.6366467

Time for backsubstitution: 5.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 439
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 3133

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2275

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400248, upper bound: 0.0400174
time: 145.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400124, upper bound: 0.0400323
time: 6.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.8308637, -2.7324901, -3.8308637, -2.7324901, -0.6773958, 0.6773970
1: -4.8651872, -3.0135963, -4.8651872, -3.0135963, -0.9633362, 0.9633358
2: -0.7754370, -0.5610399, -0.7754370, -0.5610399, -0.1354360, 0.1354367
3: 0.1295116, 0.4294550, 0.1295116, 0.4294550, -0.1016415, 0.1016443
4: -1.0549200, -0.4258140, -1.0549200, -0.4258140, -0.2309306, 0.2309236
5: 0.2908296, 0.5544203, 0.2908296, 0.5544203, -0.1145847, 0.1145873
6: -2.8310716, -1.9714756, -2.8310716, -1.9714756, -0.2252236, 0.2252251
7: 0.7645508, 1.2200117, 0.7645508, 1.2200117, -0.0829536, 0.0829534
8: -4.1439810, -2.5720947, -4.1439810, -2.5720947, -0.6319607, 0.6319579
9: -3.3517365, -1.9905157, -3.3517365, -1.9905157, -0.6366476, 0.6366493

Time for backsubstitution: 5.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 439
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2275

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 686

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400248, upper bound: 0.0400264
time: 502.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400247, upper bound: 0.0400313
time: 90.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.8308637, -2.7324901, -3.8308637, -2.7324901, -0.6773971, 0.6773961
1: -4.8651872, -3.0135963, -4.8651872, -3.0135963, -0.9633358, 0.9633362
2: -0.7754370, -0.5610399, -0.7754370, -0.5610399, -0.1354367, 0.1354359
3: 0.1295116, 0.4294550, 0.1295116, 0.4294550, -0.1016443, 0.1016416
4: -1.0549200, -0.4258140, -1.0549200, -0.4258140, -0.2309237, 0.2309306
5: 0.2908296, 0.5544203, 0.2908296, 0.5544203, -0.1145873, 0.1145847
6: -2.8310716, -1.9714756, -2.8310716, -1.9714756, -0.2252251, 0.2252236
7: 0.7645508, 1.2200117, 0.7645508, 1.2200117, -0.0829534, 0.0829536
8: -4.1439810, -2.5720947, -4.1439810, -2.5720947, -0.6319579, 0.6319607
9: -3.3517365, -1.9905157, -3.3517365, -1.9905157, -0.6366493, 0.6366476

Time for backsubstitution: 5.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 439
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 3433

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3069

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400198, upper bound: 0.0400269
time: 22.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400229, upper bound: 0.0400218
time: 203.09 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.8308637, -2.7324901, -3.8308637, -2.7324901, -0.6773263, 0.6773198
1: -4.8651872, -3.0135963, -4.8651872, -3.0135963, -0.9627854, 0.9627523
2: -0.7754370, -0.5610399, -0.7754370, -0.5610399, -0.1354285, 0.1354286
3: 0.1295116, 0.4294550, 0.1295116, 0.4294550, -0.1016383, 0.1016381
4: -1.0549200, -0.4258140, -1.0549200, -0.4258140, -0.2306659, 0.2306762
5: 0.2908296, 0.5544203, 0.2908296, 0.5544203, -0.1145843, 0.1145840
6: -2.8310716, -1.9714756, -2.8310716, -1.9714756, -0.2251242, 0.2251263
7: 0.7645508, 1.2200117, 0.7645508, 1.2200117, -0.0829532, 0.0829531
8: -4.1439810, -2.5720947, -4.1439810, -2.5720947, -0.6318519, 0.6318557
9: -3.3517365, -1.9905157, -3.3517365, -1.9905157, -0.6360706, 0.6360216

Time for backsubstitution: 5.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 439
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 3139

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 850

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400073, upper bound: 0.0400131
time: 7.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400087, upper bound: 0.0400085
time: 35.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.8308637, -2.7324901, -3.8308637, -2.7324901, -0.6773270, 0.6773190
1: -4.8651872, -3.0135963, -4.8651872, -3.0135963, -0.9627860, 0.9627519
2: -0.7754370, -0.5610399, -0.7754370, -0.5610399, -0.1354285, 0.1354286
3: 0.1295116, 0.4294550, 0.1295116, 0.4294550, -0.1016382, 0.1016382
4: -1.0549200, -0.4258140, -1.0549200, -0.4258140, -0.2306659, 0.2306762
5: 0.2908296, 0.5544203, 0.2908296, 0.5544203, -0.1145842, 0.1145841
6: -2.8310716, -1.9714756, -2.8310716, -1.9714756, -0.2251242, 0.2251263
7: 0.7645508, 1.2200117, 0.7645508, 1.2200117, -0.0829531, 0.0829532
8: -4.1439810, -2.5720947, -4.1439810, -2.5720947, -0.6318529, 0.6318547
9: -3.3517365, -1.9905157, -3.3517365, -1.9905157, -0.6360707, 0.6360214

Time for backsubstitution: 5.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 439
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 3048

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 763

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400164, upper bound: 0.0400239
time: 25.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400169, upper bound: 0.0400242
time: 53.87 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.8308637, -2.7324901, -3.8308637, -2.7324901, -0.6773251, 0.6773221
1: -4.8651872, -3.0135963, -4.8651872, -3.0135963, -0.9627559, 0.9627612
2: -0.7754370, -0.5610399, -0.7754370, -0.5610399, -0.1354288, 0.1354283
3: 0.1295116, 0.4294550, 0.1295116, 0.4294550, -0.1016408, 0.1016410
4: -1.0549200, -0.4258140, -1.0549200, -0.4258140, -0.2306766, 0.2306836
5: 0.2908296, 0.5544203, 0.2908296, 0.5544203, -0.1145842, 0.1145847
6: -2.8310716, -1.9714756, -2.8310716, -1.9714756, -0.2251365, 0.2251421
7: 0.7645508, 1.2200117, 0.7645508, 1.2200117, -0.0829534, 0.0829536
8: -4.1439810, -2.5720947, -4.1439810, -2.5720947, -0.6318603, 0.6318468
9: -3.3517365, -1.9905157, -3.3517365, -1.9905157, -0.6360354, 0.6360445

Time for backsubstitution: 5.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 439
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 3023

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2300

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400092, upper bound: 0.0400251
time: 122.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400209, upper bound: 0.0400118
time: 213.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.8308637, -2.7324901, -3.8308637, -2.7324901, -0.6773149, 0.6773322
1: -4.8651872, -3.0135963, -4.8651872, -3.0135963, -0.9627275, 0.9627893
2: -0.7754370, -0.5610399, -0.7754370, -0.5610399, -0.1354284, 0.1354288
3: 0.1295116, 0.4294550, 0.1295116, 0.4294550, -0.1016410, 0.1016409
4: -1.0549200, -0.4258140, -1.0549200, -0.4258140, -0.2306939, 0.2306663
5: 0.2908296, 0.5544203, 0.2908296, 0.5544203, -0.1145845, 0.1145844
6: -2.8310716, -1.9714756, -2.8310716, -1.9714756, -0.2251442, 0.2251344
7: 0.7645508, 1.2200117, 0.7645508, 1.2200117, -0.0829536, 0.0829534
8: -4.1439810, -2.5720947, -4.1439810, -2.5720947, -0.6318496, 0.6318575
9: -3.3517365, -1.9905157, -3.3517365, -1.9905157, -0.6359953, 0.6360844

Time for backsubstitution: 5.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 439
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 2972

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2258

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400124, upper bound: 0.0400291
time: 30.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400201, upper bound: 0.0400171
time: 166.38 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 202.93 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 202.93
Output dim: 7, lower bound: -0.0400245, upper bound: 0.0400164
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 202.93
Output dim: 7, lower bound: -0.0400099, upper bound: 0.0400280
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 202.93
Output dim: 7, lower bound: -0.0400248, upper bound: 0.0400174
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 202.93
Output dim: 7, lower bound: -0.0400124, upper bound: 0.0400323
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 202.93
Output dim: 7, lower bound: -0.0400248, upper bound: 0.0400264
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 202.93
Output dim: 7, lower bound: -0.0400247, upper bound: 0.0400313
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 202.93
Output dim: 7, lower bound: -0.0400198, upper bound: 0.0400269
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 202.93
Output dim: 7, lower bound: -0.0400229, upper bound: 0.0400218
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 202.93
Output dim: 7, lower bound: -0.0400073, upper bound: 0.0400131
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 202.93
Output dim: 7, lower bound: -0.0400087, upper bound: 0.0400085
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 202.93
Output dim: 7, lower bound: -0.0400164, upper bound: 0.0400239
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 202.93
Output dim: 7, lower bound: -0.0400169, upper bound: 0.0400242
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 202.93
Output dim: 7, lower bound: -0.0400092, upper bound: 0.0400251
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 202.93
Output dim: 7, lower bound: -0.0400209, upper bound: 0.0400118
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 202.93
Output dim: 7, lower bound: -0.0400124, upper bound: 0.0400291
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 202.93
Output dim: 7, lower bound: -0.0400201, upper bound: 0.0400171

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.8308637, -2.7324901, -3.8308637, -2.7324901, -0.6692481, 0.6687105
1: -4.8651872, -3.0135963, -4.8651872, -3.0135963, -0.9509071, 0.9502424
2: -0.7754370, -0.5610399, -0.7754370, -0.5610399, -0.1350676, 0.1350409
3: 0.1295116, 0.4294550, 0.1295116, 0.4294550, -0.1010798, 0.1011121
4: -1.0549200, -0.4258140, -1.0549200, -0.4258140, -0.2269172, 0.2270865
5: 0.2908296, 0.5544203, 0.2908296, 0.5544203, -0.1140570, 0.1140974
6: -2.8310716, -1.9714756, -2.8310716, -1.9714756, -0.2230006, 0.2230844
7: 0.7645508, 1.2200117, 0.7645508, 1.2200117, -0.0829522, 0.0829526
8: -4.1439810, -2.5720947, -4.1439810, -2.5720947, -0.6267520, 0.6264210
9: -3.3517365, -1.9905157, -3.3517365, -1.9905157, -0.6326232, 0.6323628

Time for backsubstitution: 5.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 439
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2940

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2275

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400240, upper bound: 0.0400047
time: 197.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400105, upper bound: 0.0400156
time: 22.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.8308637, -2.7324901, -3.8308637, -2.7324901, -0.6687475, 0.6692111
1: -4.8651872, -3.0135963, -4.8651872, -3.0135963, -0.9502646, 0.9508852
2: -0.7754370, -0.5610399, -0.7754370, -0.5610399, -0.1350419, 0.1350666
3: 0.1295116, 0.4294550, 0.1295116, 0.4294550, -0.1011122, 0.1010798
4: -1.0549200, -0.4258140, -1.0549200, -0.4258140, -0.2270919, 0.2269118
5: 0.2908296, 0.5544203, 0.2908296, 0.5544203, -0.1140964, 0.1140580
6: -2.8310716, -1.9714756, -2.8310716, -1.9714756, -0.2230747, 0.2230103
7: 0.7645508, 1.2200117, 0.7645508, 1.2200117, -0.0829522, 0.0829527
8: -4.1439810, -2.5720947, -4.1439810, -2.5720947, -0.6265131, 0.6266599
9: -3.3517365, -1.9905157, -3.3517365, -1.9905157, -0.6324099, 0.6325761

Time for backsubstitution: 5.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 439
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 2268

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2987

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400095, upper bound: 0.0400303
time: 130.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400096, upper bound: 0.0400270
time: 42.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.8308637, -2.7324901, -3.8308637, -2.7324901, -0.6773995, 0.6774390
1: -4.8651872, -3.0135963, -4.8651872, -3.0135963, -0.9633431, 0.9633667
2: -0.7754370, -0.5610399, -0.7754370, -0.5610399, -0.1354467, 0.1354477
3: 0.1295116, 0.4294550, 0.1295116, 0.4294550, -0.1016513, 0.1016522
4: -1.0549200, -0.4258140, -1.0549200, -0.4258140, -0.2309256, 0.2309312
5: 0.2908296, 0.5544203, 0.2908296, 0.5544203, -0.1145910, 0.1145896
6: -2.8310716, -1.9714756, -2.8310716, -1.9714756, -0.2251000, 0.2250991
7: 0.7645508, 1.2200117, 0.7645508, 1.2200117, -0.0829526, 0.0829519
8: -4.1439810, -2.5720947, -4.1439810, -2.5720947, -0.6317213, 0.6318173
9: -3.3517365, -1.9905157, -3.3517365, -1.9905157, -0.6366018, 0.6366510

Time for backsubstitution: 5.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 439
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2488

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400253, upper bound: 0.0400166
time: 62.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400244, upper bound: 0.0400146
time: 86.07 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.8308637, -2.7324901, -3.8308637, -2.7324901, -0.6774019, 0.6774366
1: -4.8651872, -3.0135963, -4.8651872, -3.0135963, -0.9633447, 0.9633653
2: -0.7754370, -0.5610399, -0.7754370, -0.5610399, -0.1354468, 0.1354476
3: 0.1295116, 0.4294550, 0.1295116, 0.4294550, -0.1016522, 0.1016513
4: -1.0549200, -0.4258140, -1.0549200, -0.4258140, -0.2309258, 0.2309310
5: 0.2908296, 0.5544203, 0.2908296, 0.5544203, -0.1145906, 0.1145900
6: -2.8310716, -1.9714756, -2.8310716, -1.9714756, -0.2251088, 0.2250903
7: 0.7645508, 1.2200117, 0.7645508, 1.2200117, -0.0829524, 0.0829521
8: -4.1439810, -2.5720947, -4.1439810, -2.5720947, -0.6317253, 0.6318133
9: -3.3517365, -1.9905157, -3.3517365, -1.9905157, -0.6366041, 0.6366488

Time for backsubstitution: 6.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 439
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2113

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3086

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0399979, upper bound: 0.0400153
time: 74.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400024, upper bound: 0.0400152
time: 20.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.8308637, -2.7324901, -3.8308637, -2.7324901, -0.6773719, 0.6773684
1: -4.8651872, -3.0135963, -4.8651872, -3.0135963, -0.9633300, 0.9633286
2: -0.7754370, -0.5610399, -0.7754370, -0.5610399, -0.1354351, 0.1354358
3: 0.1295116, 0.4294550, 0.1295116, 0.4294550, -0.1016383, 0.1016411
4: -1.0549200, -0.4258140, -1.0549200, -0.4258140, -0.2309265, 0.2309195
5: 0.2908296, 0.5544203, 0.2908296, 0.5544203, -0.1145837, 0.1145864
6: -2.8310716, -1.9714756, -2.8310716, -1.9714756, -0.2252153, 0.2252171
7: 0.7645508, 1.2200117, 0.7645508, 1.2200117, -0.0829533, 0.0829531
8: -4.1439810, -2.5720947, -4.1439810, -2.5720947, -0.6319376, 0.6319338
9: -3.3517365, -1.9905157, -3.3517365, -1.9905157, -0.6366361, 0.6366360

Time for backsubstitution: 6.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 439
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2370

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2925

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400257, upper bound: 0.0400301
time: 87.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400230, upper bound: 0.0400332
time: 20.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.8308637, -2.7324901, -3.8308637, -2.7324901, -0.6773671, 0.6773732
1: -4.8651872, -3.0135963, -4.8651872, -3.0135963, -0.9633290, 0.9633296
2: -0.7754370, -0.5610399, -0.7754370, -0.5610399, -0.1354350, 0.1354358
3: 0.1295116, 0.4294550, 0.1295116, 0.4294550, -0.1016383, 0.1016410
4: -1.0549200, -0.4258140, -1.0549200, -0.4258140, -0.2309265, 0.2309195
5: 0.2908296, 0.5544203, 0.2908296, 0.5544203, -0.1145838, 0.1145863
6: -2.8310716, -1.9714756, -2.8310716, -1.9714756, -0.2252156, 0.2252168
7: 0.7645508, 1.2200117, 0.7645508, 1.2200117, -0.0829533, 0.0829531
8: -4.1439810, -2.5720947, -4.1439810, -2.5720947, -0.6319367, 0.6319348
9: -3.3517365, -1.9905157, -3.3517365, -1.9905157, -0.6366343, 0.6366377

Time for backsubstitution: 5.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 439
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 2986

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2079

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400226, upper bound: 0.0400294
time: 100.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400243, upper bound: 0.0400283
time: 5.77 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 112.65 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 112.65
Output dim: 7, lower bound: -0.0400240, upper bound: 0.0400047
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 112.65
Output dim: 7, lower bound: -0.0400105, upper bound: 0.0400156
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 112.65
Output dim: 7, lower bound: -0.0400095, upper bound: 0.0400303
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 112.65
Output dim: 7, lower bound: -0.0400096, upper bound: 0.0400270
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 112.65
Output dim: 7, lower bound: -0.0400253, upper bound: 0.0400166
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 112.65
Output dim: 7, lower bound: -0.0400244, upper bound: 0.0400146
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 112.65
Output dim: 7, lower bound: -0.0399979, upper bound: 0.0400153
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 112.65
Output dim: 7, lower bound: -0.0400024, upper bound: 0.0400152
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 112.65
Output dim: 7, lower bound: -0.0400257, upper bound: 0.0400301
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 112.65
Output dim: 7, lower bound: -0.0400230, upper bound: 0.0400332
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 112.65
Output dim: 7, lower bound: -0.0400226, upper bound: 0.0400294
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 112.65
Output dim: 7, lower bound: -0.0400243, upper bound: 0.0400283
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 112.65
Output dim: 7, lower bound: -0.0400198, upper bound: 0.0400269
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 112.65
Output dim: 7, lower bound: -0.0400229, upper bound: 0.0400218
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 112.65
Output dim: 7, lower bound: -0.0400073, upper bound: 0.0400131
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 112.65
Output dim: 7, lower bound: -0.0400087, upper bound: 0.0400085
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 112.65
Output dim: 7, lower bound: -0.0400164, upper bound: 0.0400239
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 112.65
Output dim: 7, lower bound: -0.0400169, upper bound: 0.0400242
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 112.65
Output dim: 7, lower bound: -0.0400092, upper bound: 0.0400251
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 112.65
Output dim: 7, lower bound: -0.0400209, upper bound: 0.0400118
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 112.65
Output dim: 7, lower bound: -0.0400124, upper bound: 0.0400291
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 112.65
Output dim: 7, lower bound: -0.0400201, upper bound: 0.0400171

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 36.69 + 3607.04 = 3643.73 seconds
