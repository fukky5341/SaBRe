## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 0)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.0358463636


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-4.0919914, -3.0782702, -4.0919914, -3.0782702, -0.3686969, 0.3686969)
1: (-4.5387135, -3.2085807, -4.5387135, -3.2085807, -0.5495763, 0.5495764)
2: (-0.5995762, -0.0602678, -0.5995762, -0.0602678, -0.5393084, 0.5393084)
3: (-1.1253445, -0.7219793, -1.1253445, -0.7219793, -0.1813538, 0.1813538)
4: (-0.5022298, 0.2107685, -0.5022298, 0.2107685, -0.5793709, 0.5793708)
5: (-1.6517293, -1.2967341, -1.6517293, -1.2967341, -0.1557691, 0.1557691)
6: (0.6691452, 0.7768360, 0.6691452, 0.7768360, -0.0744242, 0.0744242)
7: (-2.2282453, -1.6877239, -2.2282453, -1.6877239, -0.2697740, 0.2697739)
8: (-4.9219270, -3.9530594, -4.9219270, -3.9530594, -0.4055363, 0.4055363)
9: (-4.4394484, -3.4853926, -4.4394484, -3.4853926, -0.3410602, 0.3410602)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.65 + 74.85 = 82.49 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0359040, upper bound: 0.0359104

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 2570
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 2584
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 2108
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 3004
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 2554
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 2404
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2555
type: A, layer: 1, pos: 338
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 2578
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 3489
type: A, layer: 1, pos: 3493
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3248
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2562
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 2671
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2553
type: A, layer: 1, pos: 3249
type: A, layer: 1, pos: 3503
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2097
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 2199
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2764
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 3127
type: A, layer: 1, pos: 3097
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 3034
type: A, layer: 1, pos: 2795
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 2765
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 322
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 2454
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 3263
type: A, layer: 1, pos: 2677
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 3488
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 2631
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 3454
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 2676
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 2551
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 2763
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 2660
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 2499
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 2452
type: A, layer: 1, pos: 2487
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2502
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 499
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 905
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 164
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1079
type: A, layer: 1, pos: 1094
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 2099
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2579
type: A, layer: 1, pos: 2686
type: A, layer: 1, pos: 3059
type: A, layer: 1, pos: 3142
type: A, layer: 1, pos: 3143

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2569

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0358792, upper bound: 0.0357418
time: 73.21 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0358830, upper bound: 0.0358914
time: 92.23 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 165.51 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 165.51
Output dim: 3, lower bound: -0.0358792, upper bound: 0.0357418
NS_A2, status: Status.UNKNOWN, split count: 1, time: 165.51
Output dim: 3, lower bound: -0.0358830, upper bound: 0.0358914

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -4.0859070, -3.0783522, -4.0865750, -3.0783439, -0.3589837, 0.3596596
1: -4.5295157, -3.2087455, -4.5304947, -3.2087288, -0.5358865, 0.5368644
2: -0.5995035, -0.0602832, -0.5995104, -0.0602816, -0.5392218, 0.5392272
3: -1.1252946, -0.7239392, -1.1252995, -0.7237244, -0.1789000, 0.1786891
4: -0.5021985, 0.2105796, -0.5022017, 0.2105993, -0.5791056, 0.5790886
5: -1.6516761, -1.2990646, -1.6516811, -1.2988131, -0.1525329, 0.1522814
6: 0.6691532, 0.7768106, 0.6691525, 0.7768133, -0.0743887, 0.0743863
7: -2.2282162, -1.6878703, -2.2282190, -1.6878548, -0.2695575, 0.2695417
8: -4.9182801, -3.9531066, -4.9186602, -3.9531021, -0.3994706, 0.3998927
9: -4.4349480, -3.4854274, -4.4354210, -3.4854238, -0.3333679, 0.3338845

Time for backsubstitution: 5.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2570
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 2585
type: B, layer: 1, pos: 2584
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 2108
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 3004
type: B, layer: 1, pos: 2554
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 2404
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2555
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2578
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 3493
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3248
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2562
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 2671
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2553
type: B, layer: 1, pos: 3249
type: B, layer: 1, pos: 3503
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2097
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2199
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2764
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 3127
type: B, layer: 1, pos: 3097
type: B, layer: 1, pos: 3034
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 2795
type: B, layer: 1, pos: 2765
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 322
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 3263
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 3488
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 2631
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 3454
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 2763
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 2660
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 2499
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 2666
type: B, layer: 1, pos: 2452
type: B, layer: 1, pos: 2487
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2502
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 499
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 164
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1079
type: B, layer: 1, pos: 1094
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 2099
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2579
type: B, layer: 1, pos: 2686
type: B, layer: 1, pos: 3059
type: B, layer: 1, pos: 3142
type: B, layer: 1, pos: 3143

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2570

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0357469, upper bound: 0.0357102
time: 62.04 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0358487, upper bound: 0.0357081
time: 80.72 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -4.0905704, -3.0661047, -4.0905967, -3.0782893, -0.3605491, 0.3773493
1: -4.5372219, -3.1899378, -4.5373759, -3.2086201, -0.5383568, 0.5628356
2: -0.6001569, -0.0602225, -0.5995316, -0.0602734, -0.5398836, 0.5393091
3: -1.1293666, -0.7226421, -1.1253233, -0.7225900, -0.1840870, 0.1790802
4: -0.5026659, 0.2105626, -0.5022105, 0.2105786, -0.5795844, 0.5790964
5: -1.6564773, -1.2975411, -1.6517104, -1.2974820, -0.1591624, 0.1527167
6: 0.6690897, 0.7768171, 0.6691501, 0.7768153, -0.0744890, 0.0744020
7: -2.2291696, -1.6877885, -2.2282305, -1.6878281, -0.2707929, 0.2696766
8: -4.9207692, -3.9457169, -4.9208279, -3.9530780, -0.4001295, 0.4111161
9: -4.4385376, -3.4762774, -4.4385891, -3.4854069, -0.3341924, 0.3474196

Time for backsubstitution: 5.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2570
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 2585
type: B, layer: 1, pos: 2584
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 2108
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 3004
type: B, layer: 1, pos: 2554
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 2404
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2555
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2578
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 3493
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3248
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2562
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 2671
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2553
type: B, layer: 1, pos: 3249
type: B, layer: 1, pos: 3503
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2097
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2199
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2764
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 3127
type: B, layer: 1, pos: 3097
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 3034
type: B, layer: 1, pos: 2795
type: B, layer: 1, pos: 2765
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 322
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 3263
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 3488
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 2631
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 3454
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 2763
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 2660
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 2499
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 2666
type: B, layer: 1, pos: 2452
type: B, layer: 1, pos: 2487
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2502
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 499
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 164
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1079
type: B, layer: 1, pos: 1094
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 2099
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2579
type: B, layer: 1, pos: 2686
type: B, layer: 1, pos: 3059
type: B, layer: 1, pos: 3142
type: B, layer: 1, pos: 3143

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2570

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0357520, upper bound: 0.0358589
time: 94.75 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0358542, upper bound: 0.0358552
time: 53.37 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 154.14 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 154.14
Output dim: 3, lower bound: -0.0357469, upper bound: 0.0357102
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 154.14
Output dim: 3, lower bound: -0.0358487, upper bound: 0.0357081
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 154.14
Output dim: 3, lower bound: -0.0357520, upper bound: 0.0358589
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 154.14
Output dim: 3, lower bound: -0.0358542, upper bound: 0.0358552

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -4.0839996, -3.0783753, -4.0845385, -3.0675135, -0.3696710, 0.3556030
1: -4.5271530, -3.2087979, -4.5279169, -3.1927109, -0.5504839, 0.5310125
2: -0.5994523, -0.0603239, -0.6001810, -0.0603121, -0.5391402, 0.5398571
3: -1.1252694, -0.7246609, -1.1291178, -0.7245132, -0.1774174, 0.1823104
4: -0.5021726, 0.2103301, -0.5028936, 0.2103256, -0.5788158, 0.5795600
5: -1.6516570, -1.2999243, -1.6561861, -1.2997723, -0.1507622, 0.1560974
6: 0.6691575, 0.7767981, 0.6691223, 0.7768008, -0.0743629, 0.0744040
7: -2.2282012, -1.6879758, -2.2289495, -1.6879592, -0.2694183, 0.2699364
8: -4.9171653, -3.9531245, -4.9174185, -3.9472973, -0.4048010, 0.3974582
9: -4.4339714, -3.4854345, -4.4343672, -3.4777851, -0.3406520, 0.3309149

Time for backsubstitution: 5.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 2584
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 2108
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 3004
type: A, layer: 1, pos: 2554
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 2404
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2555
type: A, layer: 1, pos: 338
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 2578
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 3489
type: A, layer: 1, pos: 3493
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3248
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2562
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 2671
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2553
type: A, layer: 1, pos: 3249
type: A, layer: 1, pos: 3503
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2097
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 2199
type: A, layer: 1, pos: 2570
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2764
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 3127
type: A, layer: 1, pos: 3097
type: A, layer: 1, pos: 3034
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2795
type: A, layer: 1, pos: 2765
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 322
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 2454
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 3263
type: A, layer: 1, pos: 2677
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 3488
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 2631
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 3454
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 2676
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 2551
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 2763
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 2660
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 2499
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 2452
type: A, layer: 1, pos: 2487
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2502
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 499
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 905
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 164
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1079
type: A, layer: 1, pos: 1094
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 2099
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2579
type: A, layer: 1, pos: 2686
type: A, layer: 1, pos: 3059
type: A, layer: 1, pos: 3142
type: A, layer: 1, pos: 3143

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0357119, upper bound: 0.0357085
time: 48.49 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0358483, upper bound: 0.0357117
time: 128.08 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -4.0861616, -3.0661926, -4.0856361, -3.0783963, -0.3553196, 0.3718842
1: -4.5306025, -3.1901312, -4.5299139, -3.2088380, -0.5311631, 0.5553164
2: -0.6000919, -0.0602716, -0.5994602, -0.0603292, -0.5397628, 0.5391886
3: -1.1293085, -0.7242002, -1.1252584, -0.7243345, -0.1822402, 0.1773774
4: -0.5026373, 0.2102721, -0.5021787, 0.2102571, -0.5792298, 0.5787726
5: -1.6564217, -1.2993697, -1.6516461, -1.2995284, -0.1570371, 0.1507119
6: 0.6690980, 0.7768099, 0.6691590, 0.7768071, -0.0744662, 0.0743792
7: -2.2291503, -1.6878874, -2.2282104, -1.6879377, -0.2707189, 0.2696080
8: -4.9183207, -3.9457610, -4.9180884, -3.9531281, -0.3975524, 0.4084618
9: -4.4354358, -3.4763079, -4.4350896, -3.4854414, -0.3307191, 0.3438522

Time for backsubstitution: 6.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 2584
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 2108
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 3004
type: A, layer: 1, pos: 2554
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 2404
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2555
type: A, layer: 1, pos: 338
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 2578
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 3489
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3493
type: A, layer: 1, pos: 3248
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2562
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 2671
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2553
type: A, layer: 1, pos: 3249
type: A, layer: 1, pos: 3503
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2097
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 2570
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 2199
type: A, layer: 1, pos: 2764
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 3127
type: A, layer: 1, pos: 3097
type: A, layer: 1, pos: 3034
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2795
type: A, layer: 1, pos: 2765
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 322
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 2454
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 3263
type: A, layer: 1, pos: 2677
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 3488
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 2631
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 3454
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 2676
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 2551
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 2763
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 2660
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 2499
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 2452
type: A, layer: 1, pos: 2487
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2502
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 499
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 905
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 164
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1079
type: A, layer: 1, pos: 1094
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 2099
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2579
type: A, layer: 1, pos: 2686
type: A, layer: 1, pos: 3059
type: A, layer: 1, pos: 3142
type: A, layer: 1, pos: 3143

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0356144, upper bound: 0.0358567
time: 199.23 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0357516, upper bound: 0.0358534
time: 157.27 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -4.0886602, -3.0661259, -4.0885344, -3.0674634, -0.3712391, 0.3732926
1: -4.5348015, -3.1899867, -4.5347328, -3.1925974, -0.5529187, 0.5569865
2: -0.6001042, -0.0602629, -0.6002015, -0.0603031, -0.5398011, 0.5399386
3: -1.1293418, -0.7233750, -1.1291422, -0.7234019, -0.1826044, 0.1826970
4: -0.5026391, 0.2103102, -0.5029014, 0.2103024, -0.5792905, 0.5795635
5: -1.6564586, -1.2984265, -1.6562152, -1.2984668, -0.1573946, 0.1565231
6: 0.6690941, 0.7768044, 0.6691197, 0.7768026, -0.0744629, 0.0744196
7: -2.2291534, -1.6878998, -2.2289615, -1.6879377, -0.2706524, 0.2700649
8: -4.9195852, -3.9457340, -4.9195223, -3.9472709, -0.4054314, 0.4087342
9: -4.4375033, -3.4762850, -4.4374533, -3.4777689, -0.3414608, 0.3444519

Time for backsubstitution: 5.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 2584
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 2108
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 3004
type: A, layer: 1, pos: 2554
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 2404
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2555
type: A, layer: 1, pos: 338
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 2578
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 3489
type: A, layer: 1, pos: 3493
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3248
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2562
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 2671
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2553
type: A, layer: 1, pos: 3249
type: A, layer: 1, pos: 3503
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2097
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 2199
type: A, layer: 1, pos: 2570
type: A, layer: 1, pos: 2764
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 3127
type: A, layer: 1, pos: 3097
type: A, layer: 1, pos: 3034
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2795
type: A, layer: 1, pos: 2765
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 322
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 2454
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 3263
type: A, layer: 1, pos: 2677
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 3488
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 2631
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 3454
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 2676
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 2551
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 2763
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 2660
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 2499
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 2452
type: A, layer: 1, pos: 2487
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2502
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 499
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 905
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 164
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1079
type: A, layer: 1, pos: 1094
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 2099
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2579
type: A, layer: 1, pos: 2686
type: A, layer: 1, pos: 3059
type: A, layer: 1, pos: 3142
type: A, layer: 1, pos: 3143

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0357170, upper bound: 0.0357125
time: 162.02 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0358533, upper bound: 0.0358602
time: 6.63 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 174.70 seconds
NS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 174.70
Output dim: 3, lower bound: -0.0357119, upper bound: 0.0357085
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 174.70
Output dim: 3, lower bound: -0.0358483, upper bound: 0.0357117
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 174.70
Output dim: 3, lower bound: -0.0356144, upper bound: 0.0358567
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 174.70
Output dim: 3, lower bound: -0.0357516, upper bound: 0.0358534
NS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 174.70
Output dim: 3, lower bound: -0.0357170, upper bound: 0.0357125
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 174.70
Output dim: 3, lower bound: -0.0358533, upper bound: 0.0358602

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -4.0839958, -3.0784786, -4.0845356, -3.0676196, -0.3696189, 0.3437900
1: -4.5271521, -3.2090330, -4.5279160, -3.1929533, -0.5504172, 0.5149210
2: -0.5994030, -0.0603775, -0.6001363, -0.0603604, -0.5390426, 0.5397588
3: -1.1252239, -0.7246608, -1.1290722, -0.7245133, -0.1732672, 0.1823000
4: -0.5021453, 0.2103304, -0.5028685, 0.2103260, -0.5787883, 0.5795346
5: -1.6515967, -1.2999247, -1.6561249, -1.2997725, -0.1458311, 0.1560745
6: 0.6691769, 0.7767973, 0.6691400, 0.7768002, -0.0743468, 0.0743859
7: -2.2281251, -1.6879969, -2.2288806, -1.6879787, -0.2693122, 0.2698489
8: -4.9171638, -3.9534698, -4.9174180, -3.9476330, -0.4047157, 0.3925809
9: -4.4339714, -3.4857860, -4.4343681, -3.4781017, -0.3405476, 0.3265666

Time for backsubstitution: 6.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2585
type: B, layer: 1, pos: 2584
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 2108
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 3004
type: B, layer: 1, pos: 2554
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 2404
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2555
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2578
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 3493
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 3248
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2562
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 2671
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2553
type: B, layer: 1, pos: 3249
type: B, layer: 1, pos: 3503
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2097
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2199
type: B, layer: 1, pos: 2764
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 3127
type: B, layer: 1, pos: 3097
type: B, layer: 1, pos: 3034
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 2795
type: B, layer: 1, pos: 2765
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 322
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 3263
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 3488
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 2631
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 3454
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 2763
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 2660
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 2499
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 2666
type: B, layer: 1, pos: 2452
type: B, layer: 1, pos: 2487
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2502
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 499
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 164
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1079
type: B, layer: 1, pos: 1094
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 2099
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2579
type: B, layer: 1, pos: 2686
type: B, layer: 1, pos: 3059
type: B, layer: 1, pos: 3142
type: B, layer: 1, pos: 3143

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2585

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0356900, upper bound: 0.0356656
time: 112.07 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0358167, upper bound: 0.0356822
time: 19.00 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -4.0814109, -3.0730836, -4.0856080, -3.0846143, -0.3436951, 0.3646484
1: -4.5243487, -3.1994958, -4.5299125, -3.2172942, -0.5154651, 0.5456244
2: -0.5998120, -0.0603405, -0.5992977, -0.0603913, -0.5394207, 0.5389572
3: -1.1268529, -0.7258738, -1.1230626, -0.7243345, -0.1796697, 0.1732184
4: -0.4998791, 0.2084056, -0.4998286, 0.2102503, -0.5765417, 0.5745597
5: -1.6535773, -1.3013045, -1.6490849, -1.2995293, -0.1540598, 0.1458967
6: 0.6691261, 0.7767922, 0.6691807, 0.7768052, -0.0744372, 0.0743418
7: -2.2289479, -1.6879153, -2.2280650, -1.6879630, -0.2704890, 0.2694189
8: -4.9165831, -3.9487944, -4.9180870, -3.9558344, -0.3926483, 0.4054634
9: -4.4340334, -3.4789679, -4.4350901, -3.4878511, -0.3265107, 0.3411950

Time for backsubstitution: 6.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2585
type: B, layer: 1, pos: 2584
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 2108
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 3004
type: B, layer: 1, pos: 2554
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 2404
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2555
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2578
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3493
type: B, layer: 1, pos: 3248
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2562
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 2671
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2553
type: B, layer: 1, pos: 3249
type: B, layer: 1, pos: 3503
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2097
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2199
type: B, layer: 1, pos: 2764
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 3127
type: B, layer: 1, pos: 3097
type: B, layer: 1, pos: 3034
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 2795
type: B, layer: 1, pos: 2765
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 322
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 3263
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 3488
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 2631
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 3454
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 2763
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 2660
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 2499
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 2666
type: B, layer: 1, pos: 2452
type: B, layer: 1, pos: 2487
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2502
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 499
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 164
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1079
type: B, layer: 1, pos: 1094
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 2099
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2579
type: B, layer: 1, pos: 2686
type: B, layer: 1, pos: 3059
type: B, layer: 1, pos: 3142
type: B, layer: 1, pos: 3143

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2585

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0355098, upper bound: 0.0358193
time: 4.52 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0355772, upper bound: 0.0358223
time: 44.30 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4.0861578, -3.0663042, -4.0856323, -3.0784945, -0.3552688, 0.3600802
1: -4.5306025, -3.1903808, -4.5299129, -3.2090683, -0.5310991, 0.5392374
2: -0.6000427, -0.0603251, -0.5994158, -0.0603772, -0.5396655, 0.5390908
3: -1.1292608, -0.7242002, -1.1252153, -0.7243344, -0.1780931, 0.1773671
4: -0.5026096, 0.2102718, -0.5021540, 0.2102565, -0.5792019, 0.5787477
5: -1.6563585, -1.2993701, -1.6515882, -1.2995290, -0.1521091, 0.1506888
6: 0.6691175, 0.7768092, 0.6691766, 0.7768065, -0.0744500, 0.0743611
7: -2.2290740, -1.6879086, -2.2281418, -1.6879567, -0.2706119, 0.2695204
8: -4.9183202, -3.9461088, -4.9180875, -3.9534643, -0.3974674, 0.4035858
9: -4.4354358, -3.4766593, -4.4350901, -3.4857578, -0.3306199, 0.3395067

Time for backsubstitution: 6.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2585
type: B, layer: 1, pos: 2584
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 2108
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 3004
type: B, layer: 1, pos: 2554
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 2404
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2555
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2578
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3493
type: B, layer: 1, pos: 3248
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2562
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 2671
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2553
type: B, layer: 1, pos: 3249
type: B, layer: 1, pos: 3503
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2097
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2199
type: B, layer: 1, pos: 2764
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 3127
type: B, layer: 1, pos: 3097
type: B, layer: 1, pos: 3034
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 2795
type: B, layer: 1, pos: 2765
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 322
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 3263
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 3488
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 2631
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 3454
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 2763
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 2660
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 2499
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 2666
type: B, layer: 1, pos: 2452
type: B, layer: 1, pos: 2487
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2502
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 499
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 164
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1079
type: B, layer: 1, pos: 1094
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 2099
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2579
type: B, layer: 1, pos: 2686
type: B, layer: 1, pos: 3059
type: B, layer: 1, pos: 3142
type: B, layer: 1, pos: 3143

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2585

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0356532, upper bound: 0.0358131
time: 80.55 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0357073, upper bound: 0.0358228
time: 126.48 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.0886569, -3.0662301, -4.0885305, -3.0675645, -0.3711880, 0.3614812
1: -4.5348015, -3.1902246, -4.5347323, -3.1928313, -0.5528538, 0.5408949
2: -0.6000547, -0.0603164, -0.6001565, -0.0603511, -0.5397035, 0.5398402
3: -1.1292956, -0.7233751, -1.1290981, -0.7234019, -0.1784548, 0.1826866
4: -0.5026119, 0.2103106, -0.5028765, 0.2103021, -0.5792625, 0.5795385
5: -1.6563979, -1.2984270, -1.6561559, -1.2984670, -0.1524636, 0.1565003
6: 0.6691135, 0.7768036, 0.6691372, 0.7768021, -0.0744468, 0.0744015
7: -2.2290769, -1.6879205, -2.2288923, -1.6879567, -0.2705452, 0.2699775
8: -4.9195843, -3.9460802, -4.9195213, -3.9476073, -0.4053462, 0.4038569
9: -4.4375038, -3.4766364, -4.4374533, -3.4780862, -0.3413603, 0.3401037

Time for backsubstitution: 6.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2585
type: B, layer: 1, pos: 2584
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 2108
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 3004
type: B, layer: 1, pos: 2554
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 2404
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2555
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2578
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 3493
type: B, layer: 1, pos: 3248
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2562
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 2671
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2553
type: B, layer: 1, pos: 3249
type: B, layer: 1, pos: 3503
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2097
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 2199
type: B, layer: 1, pos: 2764
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 3127
type: B, layer: 1, pos: 3097
type: B, layer: 1, pos: 3034
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 2795
type: B, layer: 1, pos: 2765
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 322
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 3263
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 3488
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 2631
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 3454
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 2763
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 2660
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 2499
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 2666
type: B, layer: 1, pos: 2452
type: B, layer: 1, pos: 2487
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2502
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 499
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 164
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1079
type: B, layer: 1, pos: 1094
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 2099
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2579
type: B, layer: 1, pos: 2686
type: B, layer: 1, pos: 3059
type: B, layer: 1, pos: 3142
type: B, layer: 1, pos: 3143

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2585

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0356988, upper bound: 0.0358231
time: 6.95 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0358173, upper bound: 0.0358237
time: 38.85 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 52.09 seconds
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 52.09
Output dim: 3, lower bound: -0.0356900, upper bound: 0.0356656
NS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 52.09
Output dim: 3, lower bound: -0.0358167, upper bound: 0.0356822
NS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 52.09
Output dim: 3, lower bound: -0.0355098, upper bound: 0.0358193
NS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 52.09
Output dim: 3, lower bound: -0.0355772, upper bound: 0.0358223
NS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 52.09
Output dim: 3, lower bound: -0.0356532, upper bound: 0.0358131
NS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 52.09
Output dim: 3, lower bound: -0.0357073, upper bound: 0.0358228
NS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 52.09
Output dim: 3, lower bound: -0.0356988, upper bound: 0.0358231
NS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 52.09
Output dim: 3, lower bound: -0.0358173, upper bound: 0.0358237

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 82.49 + 1646.27 = 1728.76 seconds
