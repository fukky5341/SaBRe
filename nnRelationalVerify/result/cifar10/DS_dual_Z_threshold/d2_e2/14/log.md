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
execution time: IAR + RelationalAnalysis = 7.90 + 28.90 = 36.80 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0400272, upper bound: 0.0400339

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 439
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2395

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400118, upper bound: 0.0400136
time: 21.20 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400095, upper bound: 0.0400175
time: 15.00 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 36.29 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 36.29
Output dim: 7, lower bound: -0.0400118, upper bound: 0.0400136
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 36.29
Output dim: 7, lower bound: -0.0400095, upper bound: 0.0400175

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -3.8308637, -2.7324901, -3.8308637, -2.7324901, -0.6773977, 0.6773980
1: -4.8651872, -3.0135963, -4.8651872, -3.0135963, -0.9633354, 0.9633355
2: -0.7754370, -0.5610399, -0.7754370, -0.5610399, -0.1354368, 0.1354368
3: 0.1295116, 0.4294550, 0.1295116, 0.4294550, -0.1016446, 0.1016446
4: -1.0549200, -0.4258140, -1.0549200, -0.4258140, -0.2309315, 0.2309315
5: 0.2908296, 0.5544203, 0.2908296, 0.5544203, -0.1145877, 0.1145877
6: -2.8310716, -1.9714756, -2.8310716, -1.9714756, -0.2252265, 0.2252262
7: 0.7645508, 1.2200117, 0.7645508, 1.2200117, -0.0829536, 0.0829536
8: -4.1439810, -2.5720947, -4.1439810, -2.5720947, -0.6319615, 0.6319617
9: -3.3517365, -1.9905157, -3.3517365, -1.9905157, -0.6366510, 0.6366507

Time for backsubstitution: 5.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 439
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3055

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400024, upper bound: 0.0400044
time: 62.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400034, upper bound: 0.0400078
time: 62.93 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -3.8308637, -2.7324901, -3.8308637, -2.7324901, -0.6773980, 0.6773980
1: -4.8651872, -3.0135963, -4.8651872, -3.0135963, -0.9633354, 0.9633353
2: -0.7754370, -0.5610399, -0.7754370, -0.5610399, -0.1354368, 0.1354368
3: 0.1295116, 0.4294550, 0.1295116, 0.4294550, -0.1016446, 0.1016446
4: -1.0549200, -0.4258140, -1.0549200, -0.4258140, -0.2309314, 0.2309315
5: 0.2908296, 0.5544203, 0.2908296, 0.5544203, -0.1145877, 0.1145877
6: -2.8310716, -1.9714756, -2.8310716, -1.9714756, -0.2252262, 0.2252265
7: 0.7645508, 1.2200117, 0.7645508, 1.2200117, -0.0829536, 0.0829536
8: -4.1439810, -2.5720947, -4.1439810, -2.5720947, -0.6319616, 0.6319615
9: -3.3517365, -1.9905157, -3.3517365, -1.9905157, -0.6366507, 0.6366509

Time for backsubstitution: 6.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 439
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3055

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400018, upper bound: 0.0400092
time: 147.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0400017, upper bound: 0.0400085
time: 118.18 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 271.92 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 271.92
Output dim: 7, lower bound: -0.0400024, upper bound: 0.0400044
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 271.92
Output dim: 7, lower bound: -0.0400034, upper bound: 0.0400078
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 271.92
Output dim: 7, lower bound: -0.0400018, upper bound: 0.0400092
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 271.92
Output dim: 7, lower bound: -0.0400017, upper bound: 0.0400085

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.8308637, -2.7324901, -3.8308637, -2.7324901, -0.6773927, 0.6773937
1: -4.8651872, -3.0135963, -4.8651872, -3.0135963, -0.9633337, 0.9633343
2: -0.7754370, -0.5610399, -0.7754370, -0.5610399, -0.1354368, 0.1354368
3: 0.1295116, 0.4294550, 0.1295116, 0.4294550, -0.1016416, 0.1016414
4: -1.0549200, -0.4258140, -1.0549200, -0.4258140, -0.2309307, 0.2309307
5: 0.2908296, 0.5544203, 0.2908296, 0.5544203, -0.1145871, 0.1145871
6: -2.8310716, -1.9714756, -2.8310716, -1.9714756, -0.2252151, 0.2252147
7: 0.7645508, 1.2200117, 0.7645508, 1.2200117, -0.0829534, 0.0829533
8: -4.1439810, -2.5720947, -4.1439810, -2.5720947, -0.6319570, 0.6319581
9: -3.3517365, -1.9905157, -3.3517365, -1.9905157, -0.6366485, 0.6366485

Time for backsubstitution: 6.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 439
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2382

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0399966, upper bound: 0.0400009
time: 182.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0399989, upper bound: 0.0400006
time: 15.36 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.8308637, -2.7324901, -3.8308637, -2.7324901, -0.6773934, 0.6773932
1: -4.8651872, -3.0135963, -4.8651872, -3.0135963, -0.9633342, 0.9633338
2: -0.7754370, -0.5610399, -0.7754370, -0.5610399, -0.1354368, 0.1354368
3: 0.1295116, 0.4294550, 0.1295116, 0.4294550, -0.1016414, 0.1016415
4: -1.0549200, -0.4258140, -1.0549200, -0.4258140, -0.2309307, 0.2309307
5: 0.2908296, 0.5544203, 0.2908296, 0.5544203, -0.1145871, 0.1145871
6: -2.8310716, -1.9714756, -2.8310716, -1.9714756, -0.2252151, 0.2252147
7: 0.7645508, 1.2200117, 0.7645508, 1.2200117, -0.0829534, 0.0829534
8: -4.1439810, -2.5720947, -4.1439810, -2.5720947, -0.6319579, 0.6319571
9: -3.3517365, -1.9905157, -3.3517365, -1.9905157, -0.6366487, 0.6366483

Time for backsubstitution: 6.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 439
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2382

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0399964, upper bound: 0.0400015
time: 30.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0399981, upper bound: 0.0400012
time: 39.00 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.8308637, -2.7324901, -3.8308637, -2.7324901, -0.6773930, 0.6773937
1: -4.8651872, -3.0135963, -4.8651872, -3.0135963, -0.9633337, 0.9633343
2: -0.7754370, -0.5610399, -0.7754370, -0.5610399, -0.1354368, 0.1354368
3: 0.1295116, 0.4294550, 0.1295116, 0.4294550, -0.1016415, 0.1016414
4: -1.0549200, -0.4258140, -1.0549200, -0.4258140, -0.2309307, 0.2309307
5: 0.2908296, 0.5544203, 0.2908296, 0.5544203, -0.1145871, 0.1145871
6: -2.8310716, -1.9714756, -2.8310716, -1.9714756, -0.2252148, 0.2252151
7: 0.7645508, 1.2200117, 0.7645508, 1.2200117, -0.0829534, 0.0829533
8: -4.1439810, -2.5720947, -4.1439810, -2.5720947, -0.6319572, 0.6319578
9: -3.3517365, -1.9905157, -3.3517365, -1.9905157, -0.6366482, 0.6366488

Time for backsubstitution: 5.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 439
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2382

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0399939, upper bound: 0.0400030
time: 75.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0399973, upper bound: 0.0399992
time: 81.70 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.8308637, -2.7324901, -3.8308637, -2.7324901, -0.6773937, 0.6773927
1: -4.8651872, -3.0135963, -4.8651872, -3.0135963, -0.9633342, 0.9633338
2: -0.7754370, -0.5610399, -0.7754370, -0.5610399, -0.1354368, 0.1354368
3: 0.1295116, 0.4294550, 0.1295116, 0.4294550, -0.1016414, 0.1016416
4: -1.0549200, -0.4258140, -1.0549200, -0.4258140, -0.2309307, 0.2309307
5: 0.2908296, 0.5544203, 0.2908296, 0.5544203, -0.1145871, 0.1145871
6: -2.8310716, -1.9714756, -2.8310716, -1.9714756, -0.2252148, 0.2252151
7: 0.7645508, 1.2200117, 0.7645508, 1.2200117, -0.0829534, 0.0829534
8: -4.1439810, -2.5720947, -4.1439810, -2.5720947, -0.6319581, 0.6319570
9: -3.3517365, -1.9905157, -3.3517365, -1.9905157, -0.6366484, 0.6366485

Time for backsubstitution: 5.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 439
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2382

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0399939, upper bound: 0.0400016
time: 122.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0399964, upper bound: 0.0400029
time: 78.10 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 205.91 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 205.91
Output dim: 7, lower bound: -0.0399966, upper bound: 0.0400009
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 205.91
Output dim: 7, lower bound: -0.0399989, upper bound: 0.0400006
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 205.91
Output dim: 7, lower bound: -0.0399964, upper bound: 0.0400015
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 205.91
Output dim: 7, lower bound: -0.0399981, upper bound: 0.0400012
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 205.91
Output dim: 7, lower bound: -0.0399939, upper bound: 0.0400030
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 205.91
Output dim: 7, lower bound: -0.0399973, upper bound: 0.0399992
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 205.91
Output dim: 7, lower bound: -0.0399939, upper bound: 0.0400016
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 205.91
Output dim: 7, lower bound: -0.0399964, upper bound: 0.0400029

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.8308637, -2.7324901, -3.8308637, -2.7324901, -0.6773926, 0.6773937
1: -4.8651872, -3.0135963, -4.8651872, -3.0135963, -0.9633335, 0.9633338
2: -0.7754370, -0.5610399, -0.7754370, -0.5610399, -0.1354368, 0.1354368
3: 0.1295116, 0.4294550, 0.1295116, 0.4294550, -0.1016416, 0.1016414
4: -1.0549200, -0.4258140, -1.0549200, -0.4258140, -0.2309307, 0.2309307
5: 0.2908296, 0.5544203, 0.2908296, 0.5544203, -0.1145871, 0.1145870
6: -2.8310716, -1.9714756, -2.8310716, -1.9714756, -0.2252150, 0.2252146
7: 0.7645508, 1.2200117, 0.7645508, 1.2200117, -0.0829534, 0.0829533
8: -4.1439810, -2.5720947, -4.1439810, -2.5720947, -0.6319569, 0.6319579
9: -3.3517365, -1.9905157, -3.3517365, -1.9905157, -0.6366484, 0.6366483

Time for backsubstitution: 5.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 439
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2383

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0399918, upper bound: 0.0400012
time: 10.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0399918, upper bound: 0.0400014
time: 10.07 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.8308637, -2.7324901, -3.8308637, -2.7324901, -0.6773926, 0.6773937
1: -4.8651872, -3.0135963, -4.8651872, -3.0135963, -0.9633335, 0.9633338
2: -0.7754370, -0.5610399, -0.7754370, -0.5610399, -0.1354368, 0.1354368
3: 0.1295116, 0.4294550, 0.1295116, 0.4294550, -0.1016416, 0.1016414
4: -1.0549200, -0.4258140, -1.0549200, -0.4258140, -0.2309307, 0.2309307
5: 0.2908296, 0.5544203, 0.2908296, 0.5544203, -0.1145871, 0.1145870
6: -2.8310716, -1.9714756, -2.8310716, -1.9714756, -0.2252149, 0.2252146
7: 0.7645508, 1.2200117, 0.7645508, 1.2200117, -0.0829534, 0.0829533
8: -4.1439810, -2.5720947, -4.1439810, -2.5720947, -0.6319569, 0.6319579
9: -3.3517365, -1.9905157, -3.3517365, -1.9905157, -0.6366484, 0.6366483

Time for backsubstitution: 5.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 439
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2383

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0399955, upper bound: 0.0399926
time: 68.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0399947, upper bound: 0.0399976
time: 121.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.8308637, -2.7324901, -3.8308637, -2.7324901, -0.6773933, 0.6773927
1: -4.8651872, -3.0135963, -4.8651872, -3.0135963, -0.9633340, 0.9633334
2: -0.7754370, -0.5610399, -0.7754370, -0.5610399, -0.1354368, 0.1354368
3: 0.1295116, 0.4294550, 0.1295116, 0.4294550, -0.1016414, 0.1016415
4: -1.0549200, -0.4258140, -1.0549200, -0.4258140, -0.2309307, 0.2309307
5: 0.2908296, 0.5544203, 0.2908296, 0.5544203, -0.1145871, 0.1145871
6: -2.8310716, -1.9714756, -2.8310716, -1.9714756, -0.2252150, 0.2252146
7: 0.7645508, 1.2200117, 0.7645508, 1.2200117, -0.0829533, 0.0829534
8: -4.1439810, -2.5720947, -4.1439810, -2.5720947, -0.6319578, 0.6319570
9: -3.3517365, -1.9905157, -3.3517365, -1.9905157, -0.6366485, 0.6366481

Time for backsubstitution: 5.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 439
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2383

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0399909, upper bound: 0.0399988
time: 172.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0399909, upper bound: 0.0399989
time: 69.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.8308637, -2.7324901, -3.8308637, -2.7324901, -0.6773933, 0.6773927
1: -4.8651872, -3.0135963, -4.8651872, -3.0135963, -0.9633340, 0.9633334
2: -0.7754370, -0.5610399, -0.7754370, -0.5610399, -0.1354368, 0.1354368
3: 0.1295116, 0.4294550, 0.1295116, 0.4294550, -0.1016414, 0.1016415
4: -1.0549200, -0.4258140, -1.0549200, -0.4258140, -0.2309307, 0.2309307
5: 0.2908296, 0.5544203, 0.2908296, 0.5544203, -0.1145870, 0.1145871
6: -2.8310716, -1.9714756, -2.8310716, -1.9714756, -0.2252149, 0.2252146
7: 0.7645508, 1.2200117, 0.7645508, 1.2200117, -0.0829533, 0.0829534
8: -4.1439810, -2.5720947, -4.1439810, -2.5720947, -0.6319578, 0.6319570
9: -3.3517365, -1.9905157, -3.3517365, -1.9905157, -0.6366485, 0.6366481

Time for backsubstitution: 5.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 439
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2383

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0399949, upper bound: 0.0399934
time: 98.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0399937, upper bound: 0.0399945
time: 21.91 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.8308637, -2.7324901, -3.8308637, -2.7324901, -0.6773928, 0.6773932
1: -4.8651872, -3.0135963, -4.8651872, -3.0135963, -0.9633335, 0.9633338
2: -0.7754370, -0.5610399, -0.7754370, -0.5610399, -0.1354368, 0.1354368
3: 0.1295116, 0.4294550, 0.1295116, 0.4294550, -0.1016415, 0.1016414
4: -1.0549200, -0.4258140, -1.0549200, -0.4258140, -0.2309307, 0.2309307
5: 0.2908296, 0.5544203, 0.2908296, 0.5544203, -0.1145871, 0.1145870
6: -2.8310716, -1.9714756, -2.8310716, -1.9714756, -0.2252146, 0.2252149
7: 0.7645508, 1.2200117, 0.7645508, 1.2200117, -0.0829534, 0.0829533
8: -4.1439810, -2.5720947, -4.1439810, -2.5720947, -0.6319571, 0.6319578
9: -3.3517365, -1.9905157, -3.3517365, -1.9905157, -0.6366481, 0.6366485

Time for backsubstitution: 5.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 439
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2383

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0399908, upper bound: 0.0400013
time: 37.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0399908, upper bound: 0.0400002
time: 38.00 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.8308637, -2.7324901, -3.8308637, -2.7324901, -0.6773928, 0.6773932
1: -4.8651872, -3.0135963, -4.8651872, -3.0135963, -0.9633335, 0.9633338
2: -0.7754370, -0.5610399, -0.7754370, -0.5610399, -0.1354368, 0.1354368
3: 0.1295116, 0.4294550, 0.1295116, 0.4294550, -0.1016415, 0.1016414
4: -1.0549200, -0.4258140, -1.0549200, -0.4258140, -0.2309307, 0.2309307
5: 0.2908296, 0.5544203, 0.2908296, 0.5544203, -0.1145871, 0.1145870
6: -2.8310716, -1.9714756, -2.8310716, -1.9714756, -0.2252146, 0.2252150
7: 0.7645508, 1.2200117, 0.7645508, 1.2200117, -0.0829534, 0.0829533
8: -4.1439810, -2.5720947, -4.1439810, -2.5720947, -0.6319571, 0.6319578
9: -3.3517365, -1.9905157, -3.3517365, -1.9905157, -0.6366481, 0.6366485

Time for backsubstitution: 5.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 439
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2383

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0399939, upper bound: 0.0400000
time: 32.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0399928, upper bound: 0.0399953
time: 26.09 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.8308637, -2.7324901, -3.8308637, -2.7324901, -0.6773936, 0.6773927
1: -4.8651872, -3.0135963, -4.8651872, -3.0135963, -0.9633340, 0.9633334
2: -0.7754370, -0.5610399, -0.7754370, -0.5610399, -0.1354368, 0.1354368
3: 0.1295116, 0.4294550, 0.1295116, 0.4294550, -0.1016414, 0.1016416
4: -1.0549200, -0.4258140, -1.0549200, -0.4258140, -0.2309307, 0.2309307
5: 0.2908296, 0.5544203, 0.2908296, 0.5544203, -0.1145870, 0.1145871
6: -2.8310716, -1.9714756, -2.8310716, -1.9714756, -0.2252146, 0.2252149
7: 0.7645508, 1.2200117, 0.7645508, 1.2200117, -0.0829533, 0.0829534
8: -4.1439810, -2.5720947, -4.1439810, -2.5720947, -0.6319580, 0.6319569
9: -3.3517365, -1.9905157, -3.3517365, -1.9905157, -0.6366482, 0.6366483

Time for backsubstitution: 5.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 439
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2383

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0399908, upper bound: 0.0399983
time: 77.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0399902, upper bound: 0.0399983
time: 168.91 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.8308637, -2.7324901, -3.8308637, -2.7324901, -0.6773936, 0.6773927
1: -4.8651872, -3.0135963, -4.8651872, -3.0135963, -0.9633340, 0.9633334
2: -0.7754370, -0.5610399, -0.7754370, -0.5610399, -0.1354368, 0.1354368
3: 0.1295116, 0.4294550, 0.1295116, 0.4294550, -0.1016414, 0.1016416
4: -1.0549200, -0.4258140, -1.0549200, -0.4258140, -0.2309307, 0.2309307
5: 0.2908296, 0.5544203, 0.2908296, 0.5544203, -0.1145870, 0.1145871
6: -2.8310716, -1.9714756, -2.8310716, -1.9714756, -0.2252146, 0.2252150
7: 0.7645508, 1.2200117, 0.7645508, 1.2200117, -0.0829533, 0.0829534
8: -4.1439810, -2.5720947, -4.1439810, -2.5720947, -0.6319580, 0.6319569
9: -3.3517365, -1.9905157, -3.3517365, -1.9905157, -0.6366482, 0.6366484

Time for backsubstitution: 5.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 439
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2383

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0399928, upper bound: 0.0399971
time: 15.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0399930, upper bound: 0.0399994
time: 14.98 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 36.84 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 36.84
Output dim: 7, lower bound: -0.0399918, upper bound: 0.0400012
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 36.84
Output dim: 7, lower bound: -0.0399918, upper bound: 0.0400014
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 36.84
Output dim: 7, lower bound: -0.0399955, upper bound: 0.0399926
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 36.84
Output dim: 7, lower bound: -0.0399947, upper bound: 0.0399976
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 36.84
Output dim: 7, lower bound: -0.0399909, upper bound: 0.0399988
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 36.84
Output dim: 7, lower bound: -0.0399909, upper bound: 0.0399989
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 36.84
Output dim: 7, lower bound: -0.0399949, upper bound: 0.0399934
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 36.84
Output dim: 7, lower bound: -0.0399937, upper bound: 0.0399945
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 36.84
Output dim: 7, lower bound: -0.0399908, upper bound: 0.0400013
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 36.84
Output dim: 7, lower bound: -0.0399908, upper bound: 0.0400002
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 36.84
Output dim: 7, lower bound: -0.0399939, upper bound: 0.0400000
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 36.84
Output dim: 7, lower bound: -0.0399928, upper bound: 0.0399953
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 36.84
Output dim: 7, lower bound: -0.0399908, upper bound: 0.0399983
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 36.84
Output dim: 7, lower bound: -0.0399902, upper bound: 0.0399983
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 36.84
Output dim: 7, lower bound: -0.0399928, upper bound: 0.0399971
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 36.84
Output dim: 7, lower bound: -0.0399930, upper bound: 0.0399994

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.8308637, -2.7324901, -3.8308637, -2.7324901, -0.6773925, 0.6773934
1: -4.8651872, -3.0135963, -4.8651872, -3.0135963, -0.9633334, 0.9633338
2: -0.7754370, -0.5610399, -0.7754370, -0.5610399, -0.1354368, 0.1354368
3: 0.1295116, 0.4294550, 0.1295116, 0.4294550, -0.1016416, 0.1016414
4: -1.0549200, -0.4258140, -1.0549200, -0.4258140, -0.2309307, 0.2309306
5: 0.2908296, 0.5544203, 0.2908296, 0.5544203, -0.1145871, 0.1145870
6: -2.8310716, -1.9714756, -2.8310716, -1.9714756, -0.2252149, 0.2252146
7: 0.7645508, 1.2200117, 0.7645508, 1.2200117, -0.0829533, 0.0829533
8: -4.1439810, -2.5720947, -4.1439810, -2.5720947, -0.6319568, 0.6319579
9: -3.3517365, -1.9905157, -3.3517365, -1.9905157, -0.6366484, 0.6366482

Time for backsubstitution: 5.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 439
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3058

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0399873, upper bound: 0.0399936
time: 11.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0399904, upper bound: 0.0399917
time: 97.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.8308637, -2.7324901, -3.8308637, -2.7324901, -0.6773925, 0.6773937
1: -4.8651872, -3.0135963, -4.8651872, -3.0135963, -0.9633332, 0.9633338
2: -0.7754370, -0.5610399, -0.7754370, -0.5610399, -0.1354368, 0.1354368
3: 0.1295116, 0.4294550, 0.1295116, 0.4294550, -0.1016416, 0.1016414
4: -1.0549200, -0.4258140, -1.0549200, -0.4258140, -0.2309307, 0.2309306
5: 0.2908296, 0.5544203, 0.2908296, 0.5544203, -0.1145871, 0.1145870
6: -2.8310716, -1.9714756, -2.8310716, -1.9714756, -0.2252149, 0.2252146
7: 0.7645508, 1.2200117, 0.7645508, 1.2200117, -0.0829533, 0.0829533
8: -4.1439810, -2.5720947, -4.1439810, -2.5720947, -0.6319568, 0.6319579
9: -3.3517365, -1.9905157, -3.3517365, -1.9905157, -0.6366484, 0.6366483

Time for backsubstitution: 6.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 439
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3058

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0399873, upper bound: 0.0399901
time: 163.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0399904, upper bound: 0.0399948
time: 9.35 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.8308637, -2.7324901, -3.8308637, -2.7324901, -0.6773925, 0.6773937
1: -4.8651872, -3.0135963, -4.8651872, -3.0135963, -0.9633332, 0.9633338
2: -0.7754370, -0.5610399, -0.7754370, -0.5610399, -0.1354368, 0.1354368
3: 0.1295116, 0.4294550, 0.1295116, 0.4294550, -0.1016416, 0.1016414
4: -1.0549200, -0.4258140, -1.0549200, -0.4258140, -0.2309307, 0.2309306
5: 0.2908296, 0.5544203, 0.2908296, 0.5544203, -0.1145871, 0.1145870
6: -2.8310716, -1.9714756, -2.8310716, -1.9714756, -0.2252149, 0.2252146
7: 0.7645508, 1.2200117, 0.7645508, 1.2200117, -0.0829533, 0.0829533
8: -4.1439810, -2.5720947, -4.1439810, -2.5720947, -0.6319568, 0.6319579
9: -3.3517365, -1.9905157, -3.3517365, -1.9905157, -0.6366483, 0.6366483

Time for backsubstitution: 6.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 439
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3058

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0399895, upper bound: 0.0399912
time: 128.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0399907, upper bound: 0.0399867
time: 247.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.8308637, -2.7324901, -3.8308637, -2.7324901, -0.6773932, 0.6773927
1: -4.8651872, -3.0135963, -4.8651872, -3.0135963, -0.9633340, 0.9633334
2: -0.7754370, -0.5610399, -0.7754370, -0.5610399, -0.1354368, 0.1354368
3: 0.1295116, 0.4294550, 0.1295116, 0.4294550, -0.1016414, 0.1016415
4: -1.0549200, -0.4258140, -1.0549200, -0.4258140, -0.2309307, 0.2309306
5: 0.2908296, 0.5544203, 0.2908296, 0.5544203, -0.1145871, 0.1145871
6: -2.8310716, -1.9714756, -2.8310716, -1.9714756, -0.2252149, 0.2252146
7: 0.7645508, 1.2200117, 0.7645508, 1.2200117, -0.0829532, 0.0829533
8: -4.1439810, -2.5720947, -4.1439810, -2.5720947, -0.6319577, 0.6319570
9: -3.3517365, -1.9905157, -3.3517365, -1.9905157, -0.6366485, 0.6366481

Time for backsubstitution: 6.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 439
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3058

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0399870, upper bound: 0.0399890
time: 143.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0399892, upper bound: 0.0399939
time: 13.00 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.8308637, -2.7324901, -3.8308637, -2.7324901, -0.6773932, 0.6773927
1: -4.8651872, -3.0135963, -4.8651872, -3.0135963, -0.9633338, 0.9633334
2: -0.7754370, -0.5610399, -0.7754370, -0.5610399, -0.1354368, 0.1354368
3: 0.1295116, 0.4294550, 0.1295116, 0.4294550, -0.1016414, 0.1016415
4: -1.0549200, -0.4258140, -1.0549200, -0.4258140, -0.2309307, 0.2309306
5: 0.2908296, 0.5544203, 0.2908296, 0.5544203, -0.1145871, 0.1145871
6: -2.8310716, -1.9714756, -2.8310716, -1.9714756, -0.2252149, 0.2252146
7: 0.7645508, 1.2200117, 0.7645508, 1.2200117, -0.0829533, 0.0829534
8: -4.1439810, -2.5720947, -4.1439810, -2.5720947, -0.6319577, 0.6319570
9: -3.3517365, -1.9905157, -3.3517365, -1.9905157, -0.6366485, 0.6366481

Time for backsubstitution: 6.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 439
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3058

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0399870, upper bound: 0.0399964
time: 138.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0399892, upper bound: 0.0399945
time: 12.99 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.8308637, -2.7324901, -3.8308637, -2.7324901, -0.6773926, 0.6773932
1: -4.8651872, -3.0135963, -4.8651872, -3.0135963, -0.9633334, 0.9633338
2: -0.7754370, -0.5610399, -0.7754370, -0.5610399, -0.1354368, 0.1354368
3: 0.1295116, 0.4294550, 0.1295116, 0.4294550, -0.1016415, 0.1016414
4: -1.0549200, -0.4258140, -1.0549200, -0.4258140, -0.2309306, 0.2309307
5: 0.2908296, 0.5544203, 0.2908296, 0.5544203, -0.1145871, 0.1145870
6: -2.8310716, -1.9714756, -2.8310716, -1.9714756, -0.2252146, 0.2252149
7: 0.7645508, 1.2200117, 0.7645508, 1.2200117, -0.0829533, 0.0829533
8: -4.1439810, -2.5720947, -4.1439810, -2.5720947, -0.6319569, 0.6319577
9: -3.3517365, -1.9905157, -3.3517365, -1.9905157, -0.6366481, 0.6366484

Time for backsubstitution: 6.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 439
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3058

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0399855, upper bound: 0.0399939
time: 300.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0399881, upper bound: 0.0399959
time: 16.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.8308637, -2.7324901, -3.8308637, -2.7324901, -0.6773926, 0.6773932
1: -4.8651872, -3.0135963, -4.8651872, -3.0135963, -0.9633332, 0.9633338
2: -0.7754370, -0.5610399, -0.7754370, -0.5610399, -0.1354368, 0.1354368
3: 0.1295116, 0.4294550, 0.1295116, 0.4294550, -0.1016415, 0.1016414
4: -1.0549200, -0.4258140, -1.0549200, -0.4258140, -0.2309307, 0.2309307
5: 0.2908296, 0.5544203, 0.2908296, 0.5544203, -0.1145871, 0.1145870
6: -2.8310716, -1.9714756, -2.8310716, -1.9714756, -0.2252145, 0.2252149
7: 0.7645508, 1.2200117, 0.7645508, 1.2200117, -0.0829533, 0.0829533
8: -4.1439810, -2.5720947, -4.1439810, -2.5720947, -0.6319569, 0.6319578
9: -3.3517365, -1.9905157, -3.3517365, -1.9905157, -0.6366481, 0.6366485

Time for backsubstitution: 6.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2426
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3003
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 3002
type: DSZ, layer: 1, pos: 2556
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2990
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2299
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2288
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2973
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3235
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2283
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 3520
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 3132
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 3052
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 2957
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 438
type: DSZ, layer: 1, pos: 439
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 795
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2245
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2268
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2280
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2311
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2933
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 2948
type: DSZ, layer: 1, pos: 2949
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2971
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3207
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3432
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3434
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3058

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0399855, upper bound: 0.0399944
time: 150.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0399881, upper bound: 0.0399962
time: 16.30 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 173.59 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 173.59
Output dim: 7, lower bound: -0.0399873, upper bound: 0.0399936
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 173.59
Output dim: 7, lower bound: -0.0399904, upper bound: 0.0399917
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 173.59
Output dim: 7, lower bound: -0.0399873, upper bound: 0.0399901
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 173.59
Output dim: 7, lower bound: -0.0399904, upper bound: 0.0399948
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 173.59
Output dim: 7, lower bound: -0.0399895, upper bound: 0.0399912
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 173.59
Output dim: 7, lower bound: -0.0399907, upper bound: 0.0399867
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 173.59
Output dim: 7, lower bound: -0.0399870, upper bound: 0.0399890
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 173.59
Output dim: 7, lower bound: -0.0399892, upper bound: 0.0399939
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 173.59
Output dim: 7, lower bound: -0.0399870, upper bound: 0.0399964
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 173.59
Output dim: 7, lower bound: -0.0399892, upper bound: 0.0399945
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 173.59
Output dim: 7, lower bound: -0.0399855, upper bound: 0.0399939
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 173.59
Output dim: 7, lower bound: -0.0399881, upper bound: 0.0399959
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 173.59
Output dim: 7, lower bound: -0.0399855, upper bound: 0.0399944
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 173.59
Output dim: 7, lower bound: -0.0399881, upper bound: 0.0399962
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 173.59
Output dim: 7, lower bound: -0.0399939, upper bound: 0.0400000
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 173.59
Output dim: 7, lower bound: -0.0399908, upper bound: 0.0399983
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 173.59
Output dim: 7, lower bound: -0.0399902, upper bound: 0.0399983
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 173.59
Output dim: 7, lower bound: -0.0399928, upper bound: 0.0399971
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 173.59
Output dim: 7, lower bound: -0.0399930, upper bound: 0.0399994

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 36.80 + 3613.43 = 3650.23 seconds
