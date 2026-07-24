## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 14)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.0285688656


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3.7387586, -2.7396324, -3.7387586, -2.7396324, -0.4873917, 0.4873917)
1: (-4.5826964, -3.2502704, -4.5826964, -3.2502704, -0.5611193, 0.5611194)
2: (-0.0731962, 0.1371967, -0.0731962, 0.1371967, -0.1172676, 0.1172677)
3: (-1.4051898, -1.0317664, -1.4051898, -1.0317664, -0.1892098, 0.1892098)
4: (0.1325119, 0.4152871, 0.1325119, 0.4152871, -0.2587545, 0.2587545)
5: (-1.2825830, -0.9403721, -1.2825830, -0.9403721, -0.1286408, 0.1286408)
6: (0.2598491, 0.4553727, 0.2598491, 0.4553727, -0.1321599, 0.1321599)
7: (-0.7890531, 0.0075264, -0.7890531, 0.0075264, -0.5405351, 0.5405351)
8: (-5.0408831, -4.2004170, -5.0408831, -4.2004170, -0.4289716, 0.4289715)
9: (-4.4275093, -3.6195529, -4.4275093, -3.6195529, -0.3602765, 0.3602763)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 8.28 + 203.49 = 211.78 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0286829, upper bound: 0.0286831

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2754
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2779
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 2803
type: DSZ, layer: 1, pos: 2909
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2857
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 2773
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2733
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2924
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 2806
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2918
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2916
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 2750
type: DSZ, layer: 1, pos: 2902
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2860
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2917
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 3218
type: DSZ, layer: 1, pos: 2821
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2774
type: DSZ, layer: 1, pos: 2594
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2784
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2768
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2811
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 2790
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2818
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 3315
type: DSZ, layer: 1, pos: 2785
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2752
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2759
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2875
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 2757
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2758
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2741
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2817
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 2777
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2905

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2754

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286702, upper bound: 0.0286830
time: 3.60 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286814, upper bound: 0.0286708
time: 4.29 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 7.91 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 7.91
Output dim: 6, lower bound: -0.0286702, upper bound: 0.0286830
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 7.91
Output dim: 6, lower bound: -0.0286814, upper bound: 0.0286708

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -3.7387586, -2.7396324, -3.7387586, -2.7396324, -0.4869779, 0.4869539
1: -4.5826964, -3.2502704, -4.5826964, -3.2502704, -0.5591298, 0.5590613
2: -0.0731962, 0.1371967, -0.0731962, 0.1371967, -0.1170778, 0.1170867
3: -1.4051898, -1.0317664, -1.4051898, -1.0317664, -0.1891331, 0.1891311
4: 0.1325119, 0.4152871, 0.1325119, 0.4152871, -0.2585851, 0.2585921
5: -1.2825830, -0.9403721, -1.2825830, -0.9403721, -0.1285921, 0.1285934
6: 0.2598491, 0.4553727, 0.2598491, 0.4553727, -0.1318901, 0.1318974
7: -0.7890531, 0.0075264, -0.7890531, 0.0075264, -0.5404376, 0.5404236
8: -5.0408831, -4.2004170, -5.0408831, -4.2004170, -0.4283632, 0.4283375
9: -4.4275093, -3.6195529, -4.4275093, -3.6195529, -0.3586982, 0.3586359

Time for backsubstitution: 6.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2917
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2779
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2918
type: DSZ, layer: 1, pos: 3218
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 2905
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2909
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2811
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2916
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2757
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2857
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2860
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2817
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 2594
type: DSZ, layer: 1, pos: 2750
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 2777
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2758
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2784
type: DSZ, layer: 1, pos: 2741
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2773
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 2924
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2790
type: DSZ, layer: 1, pos: 3315
type: DSZ, layer: 1, pos: 2752
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2875
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2803
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 2759
type: DSZ, layer: 1, pos: 2768
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2821
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2774
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2733
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2785
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2806
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2818
type: DSZ, layer: 1, pos: 2902
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 3113

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2917

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286703, upper bound: 0.0286697
time: 182.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286703, upper bound: 0.0286827
time: 3.97 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -3.7387586, -2.7396324, -3.7387586, -2.7396324, -0.4869539, 0.4869778
1: -4.5826964, -3.2502704, -4.5826964, -3.2502704, -0.5590613, 0.5591298
2: -0.0731962, 0.1371967, -0.0731962, 0.1371967, -0.1170867, 0.1170778
3: -1.4051898, -1.0317664, -1.4051898, -1.0317664, -0.1891311, 0.1891331
4: 0.1325119, 0.4152871, 0.1325119, 0.4152871, -0.2585921, 0.2585851
5: -1.2825830, -0.9403721, -1.2825830, -0.9403721, -0.1285934, 0.1285921
6: 0.2598491, 0.4553727, 0.2598491, 0.4553727, -0.1318974, 0.1318900
7: -0.7890531, 0.0075264, -0.7890531, 0.0075264, -0.5404236, 0.5404376
8: -5.0408831, -4.2004170, -5.0408831, -4.2004170, -0.4283376, 0.4283633
9: -4.4275093, -3.6195529, -4.4275093, -3.6195529, -0.3586359, 0.3586982

Time for backsubstitution: 6.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 3218
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2806
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 2916
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 2768
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2924
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2774
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2918
type: DSZ, layer: 1, pos: 3315
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2777
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2902
type: DSZ, layer: 1, pos: 2875
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2818
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2857
type: DSZ, layer: 1, pos: 2594
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2905
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 2784
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 2860
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2733
type: DSZ, layer: 1, pos: 2759
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 2773
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2917
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2758
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2803
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2817
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2757
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2750
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2752
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 2785
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 2909
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2741
type: DSZ, layer: 1, pos: 2821
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2779
type: DSZ, layer: 1, pos: 2811
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2790
type: DSZ, layer: 1, pos: 2834

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2345

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286214, upper bound: 0.0286682
time: 85.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286793, upper bound: 0.0286127
time: 3.58 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 95.04 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 95.04
Output dim: 6, lower bound: -0.0286703, upper bound: 0.0286697
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 95.04
Output dim: 6, lower bound: -0.0286703, upper bound: 0.0286827
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 95.04
Output dim: 6, lower bound: -0.0286214, upper bound: 0.0286682
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 95.04
Output dim: 6, lower bound: -0.0286793, upper bound: 0.0286127

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.7387586, -2.7396324, -3.7387586, -2.7396324, -0.4869779, 0.4869539
1: -4.5826964, -3.2502704, -4.5826964, -3.2502704, -0.5591298, 0.5590613
2: -0.0731962, 0.1371967, -0.0731962, 0.1371967, -0.1170778, 0.1170867
3: -1.4051898, -1.0317664, -1.4051898, -1.0317664, -0.1891331, 0.1891311
4: 0.1325119, 0.4152871, 0.1325119, 0.4152871, -0.2585851, 0.2585921
5: -1.2825830, -0.9403721, -1.2825830, -0.9403721, -0.1285921, 0.1285934
6: 0.2598491, 0.4553727, 0.2598491, 0.4553727, -0.1318901, 0.1318974
7: -0.7890531, 0.0075264, -0.7890531, 0.0075264, -0.5404376, 0.5404236
8: -5.0408831, -4.2004170, -5.0408831, -4.2004170, -0.4283632, 0.4283375
9: -4.4275093, -3.6195529, -4.4275093, -3.6195529, -0.3586982, 0.3586359

Time for backsubstitution: 6.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 3315
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2774
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 2594
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2803
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2817
type: DSZ, layer: 1, pos: 2779
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2752
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2750
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2759
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 3218
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2909
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2758
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2875
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2918
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 2773
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2777
type: DSZ, layer: 1, pos: 2860
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2857
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2924
type: DSZ, layer: 1, pos: 2768
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2741
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2757
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2806
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2790
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2733
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2785
type: DSZ, layer: 1, pos: 2811
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2818
type: DSZ, layer: 1, pos: 2902
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 2821
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2905
type: DSZ, layer: 1, pos: 2784
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2916
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2893

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2876

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286685, upper bound: 0.0286797
time: 11.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286685, upper bound: 0.0286791
time: 24.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.7387586, -2.7396324, -3.7387586, -2.7396324, -0.4869779, 0.4869539
1: -4.5826964, -3.2502704, -4.5826964, -3.2502704, -0.5591298, 0.5590613
2: -0.0731962, 0.1371967, -0.0731962, 0.1371967, -0.1170778, 0.1170867
3: -1.4051898, -1.0317664, -1.4051898, -1.0317664, -0.1891331, 0.1891311
4: 0.1325119, 0.4152871, 0.1325119, 0.4152871, -0.2585851, 0.2585921
5: -1.2825830, -0.9403721, -1.2825830, -0.9403721, -0.1285921, 0.1285934
6: 0.2598491, 0.4553727, 0.2598491, 0.4553727, -0.1318901, 0.1318974
7: -0.7890531, 0.0075264, -0.7890531, 0.0075264, -0.5404376, 0.5404236
8: -5.0408831, -4.2004170, -5.0408831, -4.2004170, -0.4283632, 0.4283375
9: -4.4275093, -3.6195529, -4.4275093, -3.6195529, -0.3586982, 0.3586359

Time for backsubstitution: 6.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 2785
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2779
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 2924
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2777
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2818
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2902
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2790
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 2821
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2750
type: DSZ, layer: 1, pos: 2752
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 2768
type: DSZ, layer: 1, pos: 2918
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2803
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2817
type: DSZ, layer: 1, pos: 2759
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2909
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2860
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 3218
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2733
type: DSZ, layer: 1, pos: 2774
type: DSZ, layer: 1, pos: 2784
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2875
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2905
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2773
type: DSZ, layer: 1, pos: 2594
type: DSZ, layer: 1, pos: 2806
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 3315
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2758
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 2916
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2757
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 2811
type: DSZ, layer: 1, pos: 2857
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2741
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2869

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2958

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286677, upper bound: 0.0286825
time: 5.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286697, upper bound: 0.0286794
time: 123.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.7387586, -2.7396324, -3.7387586, -2.7396324, -0.4813936, 0.4812935
1: -4.5826964, -3.2502704, -4.5826964, -3.2502704, -0.5494955, 0.5493580
2: -0.0731962, 0.1371967, -0.0731962, 0.1371967, -0.1170607, 0.1170447
3: -1.4051898, -1.0317664, -1.4051898, -1.0317664, -0.1873547, 0.1873729
4: 0.1325119, 0.4152871, 0.1325119, 0.4152871, -0.2585812, 0.2585754
5: -1.2825830, -0.9403721, -1.2825830, -0.9403721, -0.1268062, 0.1267949
6: 0.2598491, 0.4553727, 0.2598491, 0.4553727, -0.1318824, 0.1318761
7: -0.7890531, 0.0075264, -0.7890531, 0.0075264, -0.5404382, 0.5404509
8: -5.0408831, -4.2004170, -5.0408831, -4.2004170, -0.4263434, 0.4263338
9: -4.4275093, -3.6195529, -4.4275093, -3.6195529, -0.3553873, 0.3553762

Time for backsubstitution: 6.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2774
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2733
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 2860
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2916
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 2905
type: DSZ, layer: 1, pos: 2741
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3315
type: DSZ, layer: 1, pos: 2784
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2773
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2811
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 2594
type: DSZ, layer: 1, pos: 2806
type: DSZ, layer: 1, pos: 2909
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2924
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2790
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 2752
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2779
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 3218
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 2875
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2918
type: DSZ, layer: 1, pos: 2777
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 2750
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2785
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 2768
type: DSZ, layer: 1, pos: 2818
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 2757
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2821
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 2758
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2917
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2817
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2803
type: DSZ, layer: 1, pos: 2759
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2902
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2857
type: DSZ, layer: 1, pos: 3116

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2761

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286207, upper bound: 0.0286684
time: 50.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286208, upper bound: 0.0286674
time: 114.13 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.7387586, -2.7396324, -3.7387586, -2.7396324, -0.4812696, 0.4814175
1: -4.5826964, -3.2502704, -4.5826964, -3.2502704, -0.5492896, 0.5495639
2: -0.0731962, 0.1371967, -0.0731962, 0.1371967, -0.1170536, 0.1170518
3: -1.4051898, -1.0317664, -1.4051898, -1.0317664, -0.1873709, 0.1873566
4: 0.1325119, 0.4152871, 0.1325119, 0.4152871, -0.2585824, 0.2585741
5: -1.2825830, -0.9403721, -1.2825830, -0.9403721, -0.1267962, 0.1268049
6: 0.2598491, 0.4553727, 0.2598491, 0.4553727, -0.1318835, 0.1318750
7: -0.7890531, 0.0075264, -0.7890531, 0.0075264, -0.5404368, 0.5404522
8: -5.0408831, -4.2004170, -5.0408831, -4.2004170, -0.4263083, 0.4263690
9: -4.4275093, -3.6195529, -4.4275093, -3.6195529, -0.3553138, 0.3554497

Time for backsubstitution: 6.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 3315
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 2758
type: DSZ, layer: 1, pos: 2774
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2785
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2918
type: DSZ, layer: 1, pos: 2759
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2821
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2777
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2773
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2803
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2752
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2875
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2779
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2733
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 2750
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2916
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2817
type: DSZ, layer: 1, pos: 2902
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 2924
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2741
type: DSZ, layer: 1, pos: 2909
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2790
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2818
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2917
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 2806
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2860
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2905
type: DSZ, layer: 1, pos: 2757
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 3218
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 2768
type: DSZ, layer: 1, pos: 2857
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2811
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 2784
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2594
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 3116

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2816

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286714, upper bound: 0.0286104
time: 3.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286774, upper bound: 0.0286023
time: 465.31 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 474.34 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 474.34
Output dim: 6, lower bound: -0.0286685, upper bound: 0.0286797
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 474.34
Output dim: 6, lower bound: -0.0286685, upper bound: 0.0286791
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 474.34
Output dim: 6, lower bound: -0.0286677, upper bound: 0.0286825
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 474.34
Output dim: 6, lower bound: -0.0286697, upper bound: 0.0286794
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 474.34
Output dim: 6, lower bound: -0.0286207, upper bound: 0.0286684
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 474.34
Output dim: 6, lower bound: -0.0286208, upper bound: 0.0286674
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 474.34
Output dim: 6, lower bound: -0.0286714, upper bound: 0.0286104
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 474.34
Output dim: 6, lower bound: -0.0286774, upper bound: 0.0286023

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.7387586, -2.7396324, -3.7387586, -2.7396324, -0.4862263, 0.4861789
1: -4.5826964, -3.2502704, -4.5826964, -3.2502704, -0.5576811, 0.5574682
2: -0.0731962, 0.1371967, -0.0731962, 0.1371967, -0.1169408, 0.1169368
3: -1.4051898, -1.0317664, -1.4051898, -1.0317664, -0.1889551, 0.1889325
4: 0.1325119, 0.4152871, 0.1325119, 0.4152871, -0.2585624, 0.2585711
5: -1.2825830, -0.9403721, -1.2825830, -0.9403721, -0.1284180, 0.1284028
6: 0.2598491, 0.4553727, 0.2598491, 0.4553727, -0.1318670, 0.1318724
7: -0.7890531, 0.0075264, -0.7890531, 0.0075264, -0.5402417, 0.5402441
8: -5.0408831, -4.2004170, -5.0408831, -4.2004170, -0.4268457, 0.4266657
9: -4.4275093, -3.6195529, -4.4275093, -3.6195529, -0.3573322, 0.3571327

Time for backsubstitution: 6.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2860
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 2902
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 3218
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 2773
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2857
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 2916
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2818
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2806
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 2777
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2733
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2909
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 2821
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 2785
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2918
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2750
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 2757
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2784
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2905
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2768
type: DSZ, layer: 1, pos: 2758
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 2752
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2779
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2594
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 2803
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2875
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2811
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 2759
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2790
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2817
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 2924
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 3315
type: DSZ, layer: 1, pos: 2774
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2741

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2745

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286624, upper bound: 0.0286798
time: 3.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286683, upper bound: 0.0286732
time: 65.94 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.7387586, -2.7396324, -3.7387586, -2.7396324, -0.4862027, 0.4862025
1: -4.5826964, -3.2502704, -4.5826964, -3.2502704, -0.5575368, 0.5576125
2: -0.0731962, 0.1371967, -0.0731962, 0.1371967, -0.1169279, 0.1169497
3: -1.4051898, -1.0317664, -1.4051898, -1.0317664, -0.1889344, 0.1889531
4: 0.1325119, 0.4152871, 0.1325119, 0.4152871, -0.2585641, 0.2585694
5: -1.2825830, -0.9403721, -1.2825830, -0.9403721, -0.1284015, 0.1284193
6: 0.2598491, 0.4553727, 0.2598491, 0.4553727, -0.1318650, 0.1318744
7: -0.7890531, 0.0075264, -0.7890531, 0.0075264, -0.5402582, 0.5402278
8: -5.0408831, -4.2004170, -5.0408831, -4.2004170, -0.4266914, 0.4268200
9: -4.4275093, -3.6195529, -4.4275093, -3.6195529, -0.3571950, 0.3572699

Time for backsubstitution: 6.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 2857
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2759
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 3315
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2806
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 2821
type: DSZ, layer: 1, pos: 2750
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2779
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2757
type: DSZ, layer: 1, pos: 2916
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2818
type: DSZ, layer: 1, pos: 2790
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 2785
type: DSZ, layer: 1, pos: 2784
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2924
type: DSZ, layer: 1, pos: 2752
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 3218
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2817
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2811
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2741
type: DSZ, layer: 1, pos: 2875
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2902
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2958
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2803
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2918
type: DSZ, layer: 1, pos: 2758
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2777
type: DSZ, layer: 1, pos: 2909
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2774
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 2594
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2860
type: DSZ, layer: 1, pos: 2773
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2905
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2768
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2733
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 323

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2756

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286513, upper bound: 0.0286801
time: 38.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286679, upper bound: 0.0286628
time: 24.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.7387586, -2.7396324, -3.7387586, -2.7396324, -0.4869461, 0.4869176
1: -4.5826964, -3.2502704, -4.5826964, -3.2502704, -0.5591103, 0.5590464
2: -0.0731962, 0.1371967, -0.0731962, 0.1371967, -0.1170773, 0.1170863
3: -1.4051898, -1.0317664, -1.4051898, -1.0317664, -0.1891316, 0.1891276
4: 0.1325119, 0.4152871, 0.1325119, 0.4152871, -0.2585830, 0.2585910
5: -1.2825830, -0.9403721, -1.2825830, -0.9403721, -0.1285919, 0.1285922
6: 0.2598491, 0.4553727, 0.2598491, 0.4553727, -0.1318888, 0.1318971
7: -0.7890531, 0.0075264, -0.7890531, 0.0075264, -0.5404338, 0.5404183
8: -5.0408831, -4.2004170, -5.0408831, -4.2004170, -0.4283541, 0.4282945
9: -4.4275093, -3.6195529, -4.4275093, -3.6195529, -0.3586968, 0.3586280

Time for backsubstitution: 6.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2862
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2751
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3351
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2864
type: DSZ, layer: 1, pos: 2811
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2773
type: DSZ, layer: 1, pos: 2922
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2849
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2909
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2774
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2857
type: DSZ, layer: 1, pos: 2878
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2906
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2758
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2894
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2921
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 2876
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2868
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 323
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 2821
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2918
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 2752
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 791
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2875
type: DSZ, layer: 1, pos: 2861
type: DSZ, layer: 1, pos: 2924
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2803
type: DSZ, layer: 1, pos: 3558
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2908
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2780
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2905
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2733
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2785
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2870
type: DSZ, layer: 1, pos: 2902
type: DSZ, layer: 1, pos: 2356
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2847
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2768
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2848
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 2755
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 2846
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2741
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2777
type: DSZ, layer: 1, pos: 2923
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2784
type: DSZ, layer: 1, pos: 2915
type: DSZ, layer: 1, pos: 2818
type: DSZ, layer: 1, pos: 2750
type: DSZ, layer: 1, pos: 3336
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 3218
type: DSZ, layer: 1, pos: 2806
type: DSZ, layer: 1, pos: 2794
type: DSZ, layer: 1, pos: 2812
type: DSZ, layer: 1, pos: 2869
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2594
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2904
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 2860
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2756
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2998
type: DSZ, layer: 1, pos: 2779
type: DSZ, layer: 1, pos: 2886
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2916
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 2816
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2903
type: DSZ, layer: 1, pos: 2767
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 2759
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 2757
type: DSZ, layer: 1, pos: 2790
type: DSZ, layer: 1, pos: 2817
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 3315
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2789
type: DSZ, layer: 1, pos: 2888
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 633

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2919

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286673, upper bound: 0.0286820
time: 3.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0286675, upper bound: 0.0286788
time: 351.87 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 361.68 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 361.68
Output dim: 6, lower bound: -0.0286624, upper bound: 0.0286798
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 361.68
Output dim: 6, lower bound: -0.0286683, upper bound: 0.0286732
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 361.68
Output dim: 6, lower bound: -0.0286513, upper bound: 0.0286801
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 361.68
Output dim: 6, lower bound: -0.0286679, upper bound: 0.0286628
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 361.68
Output dim: 6, lower bound: -0.0286673, upper bound: 0.0286820
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 361.68
Output dim: 6, lower bound: -0.0286675, upper bound: 0.0286788
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 361.68
Output dim: 6, lower bound: -0.0286697, upper bound: 0.0286794
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 361.68
Output dim: 6, lower bound: -0.0286207, upper bound: 0.0286684
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 361.68
Output dim: 6, lower bound: -0.0286208, upper bound: 0.0286674
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 361.68
Output dim: 6, lower bound: -0.0286714, upper bound: 0.0286104
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 361.68
Output dim: 6, lower bound: -0.0286774, upper bound: 0.0286023

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 211.78 + 1625.01 = 1836.78 seconds
