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
execution time: IAR + RelationalAnalysis = 7.66 + 74.75 = 82.42 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0359040, upper bound: 0.0359104

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 322
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3454
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 921
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 1079
type: DSZ, layer: 1, pos: 1094
type: DSZ, layer: 1, pos: 1109
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3248
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 3503
type: DSZ, layer: 1, pos: 3504

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 322

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0359032, upper bound: 0.0359088
time: 215.67 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0359053, upper bound: 0.0359114
time: 113.94 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 329.70 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 329.70
Output dim: 3, lower bound: -0.0359032, upper bound: 0.0359088
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 329.70
Output dim: 3, lower bound: -0.0359053, upper bound: 0.0359114

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -4.0919914, -3.0782702, -4.0919914, -3.0782702, -0.3664084, 0.3663904
1: -4.5387135, -3.2085807, -4.5387135, -3.2085807, -0.5405510, 0.5402838
2: -0.5995762, -0.0602678, -0.5995762, -0.0602678, -0.5393084, 0.5393084
3: -1.1253445, -0.7219793, -1.1253445, -0.7219793, -0.1810233, 0.1810001
4: -0.5022298, 0.2107685, -0.5022298, 0.2107685, -0.5793082, 0.5793092
5: -1.6517293, -1.2967341, -1.6517293, -1.2967341, -0.1529206, 0.1526992
6: 0.6691452, 0.7768360, 0.6691452, 0.7768360, -0.0730058, 0.0731179
7: -2.2282453, -1.6877239, -2.2282453, -1.6877239, -0.2648380, 0.2644291
8: -4.9219270, -3.9530594, -4.9219270, -3.9530594, -0.4053758, 0.4053844
9: -4.4394484, -3.4853926, -4.4394484, -3.4853926, -0.3407768, 0.3407845

Time for backsubstitution: 5.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3454
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 921
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 1079
type: DSZ, layer: 1, pos: 1094
type: DSZ, layer: 1, pos: 1109
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3248
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 3503
type: DSZ, layer: 1, pos: 3504

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3034

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0358044, upper bound: 0.0358634
time: 6.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0358523, upper bound: 0.0358081
time: 235.18 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -4.0919914, -3.0782702, -4.0919914, -3.0782702, -0.3663904, 0.3664084
1: -4.5387135, -3.2085807, -4.5387135, -3.2085807, -0.5402839, 0.5405511
2: -0.5995762, -0.0602678, -0.5995762, -0.0602678, -0.5393084, 0.5393084
3: -1.1253445, -0.7219793, -1.1253445, -0.7219793, -0.1810001, 0.1810233
4: -0.5022298, 0.2107685, -0.5022298, 0.2107685, -0.5793092, 0.5793081
5: -1.6517293, -1.2967341, -1.6517293, -1.2967341, -0.1526992, 0.1529206
6: 0.6691452, 0.7768360, 0.6691452, 0.7768360, -0.0731179, 0.0730058
7: -2.2282453, -1.6877239, -2.2282453, -1.6877239, -0.2644291, 0.2648380
8: -4.9219270, -3.9530594, -4.9219270, -3.9530594, -0.4053843, 0.4053758
9: -4.4394484, -3.4853926, -4.4394484, -3.4853926, -0.3407845, 0.3407769

Time for backsubstitution: 5.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3454
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 921
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 1079
type: DSZ, layer: 1, pos: 1094
type: DSZ, layer: 1, pos: 1109
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3248
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 3503
type: DSZ, layer: 1, pos: 3504

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3034

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0358046, upper bound: 0.0358604
time: 7.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0358493, upper bound: 0.0358118
time: 71.25 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 85.05 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 85.05
Output dim: 3, lower bound: -0.0358044, upper bound: 0.0358634
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 85.05
Output dim: 3, lower bound: -0.0358523, upper bound: 0.0358081
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 85.05
Output dim: 3, lower bound: -0.0358046, upper bound: 0.0358604
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 85.05
Output dim: 3, lower bound: -0.0358493, upper bound: 0.0358118

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.0919914, -3.0782702, -4.0919914, -3.0782702, -0.3664466, 0.3663214
1: -4.5387135, -3.2085807, -4.5387135, -3.2085807, -0.5406862, 0.5401992
2: -0.5995762, -0.0602678, -0.5995762, -0.0602678, -0.5393084, 0.5393084
3: -1.1253445, -0.7219793, -1.1253445, -0.7219793, -0.1810045, 0.1810522
4: -0.5022298, 0.2107685, -0.5022298, 0.2107685, -0.5793081, 0.5793092
5: -1.6517293, -1.2967341, -1.6517293, -1.2967341, -0.1528978, 0.1527301
6: 0.6691452, 0.7768360, 0.6691452, 0.7768360, -0.0730057, 0.0731180
7: -2.2282453, -1.6877239, -2.2282453, -1.6877239, -0.2648378, 0.2644288
8: -4.9219270, -3.9530594, -4.9219270, -3.9530594, -0.4053699, 0.4053394
9: -4.4394484, -3.4853926, -4.4394484, -3.4853926, -0.3408744, 0.3407378

Time for backsubstitution: 5.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3454
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 921
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 1079
type: DSZ, layer: 1, pos: 1094
type: DSZ, layer: 1, pos: 1109
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3248
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 3503
type: DSZ, layer: 1, pos: 3504

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 576

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0357947, upper bound: 0.0358517
time: 133.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0358016, upper bound: 0.0358459
time: 224.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.0919914, -3.0782702, -4.0919914, -3.0782702, -0.3663394, 0.3663904
1: -4.5387135, -3.2085807, -4.5387135, -3.2085807, -0.5404664, 0.5402838
2: -0.5995762, -0.0602678, -0.5995762, -0.0602678, -0.5393084, 0.5393084
3: -1.1253445, -0.7219793, -1.1253445, -0.7219793, -0.1810233, 0.1809812
4: -0.5022298, 0.2107685, -0.5022298, 0.2107685, -0.5793081, 0.5793092
5: -1.6517293, -1.2967341, -1.6517293, -1.2967341, -0.1529206, 0.1526765
6: 0.6691452, 0.7768360, 0.6691452, 0.7768360, -0.0730058, 0.0731178
7: -2.2282453, -1.6877239, -2.2282453, -1.6877239, -0.2648380, 0.2644288
8: -4.9219270, -3.9530594, -4.9219270, -3.9530594, -0.4053309, 0.4053844
9: -4.4394484, -3.4853926, -4.4394484, -3.4853926, -0.3407301, 0.3407845

Time for backsubstitution: 5.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3454
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 921
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 1079
type: DSZ, layer: 1, pos: 1094
type: DSZ, layer: 1, pos: 1109
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3248
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 3503
type: DSZ, layer: 1, pos: 3504

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 576

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0358399, upper bound: 0.0358092
time: 38.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0358505, upper bound: 0.0357987
time: 67.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.0919914, -3.0782702, -4.0919914, -3.0782702, -0.3664287, 0.3663394
1: -4.5387135, -3.2085807, -4.5387135, -3.2085807, -0.5404191, 0.5404664
2: -0.5995762, -0.0602678, -0.5995762, -0.0602678, -0.5393084, 0.5393084
3: -1.1253445, -0.7219793, -1.1253445, -0.7219793, -0.1809812, 0.1810754
4: -0.5022298, 0.2107685, -0.5022298, 0.2107685, -0.5793092, 0.5793081
5: -1.6517293, -1.2967341, -1.6517293, -1.2967341, -0.1526765, 0.1529515
6: 0.6691452, 0.7768360, 0.6691452, 0.7768360, -0.0731178, 0.0730059
7: -2.2282453, -1.6877239, -2.2282453, -1.6877239, -0.2644289, 0.2648377
8: -4.9219270, -3.9530594, -4.9219270, -3.9530594, -0.4053785, 0.4053308
9: -4.4394484, -3.4853926, -4.4394484, -3.4853926, -0.3408821, 0.3407301

Time for backsubstitution: 5.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3454
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 921
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 1079
type: DSZ, layer: 1, pos: 1094
type: DSZ, layer: 1, pos: 1109
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3248
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 3503
type: DSZ, layer: 1, pos: 3504

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 576

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0357958, upper bound: 0.0358610
time: 4.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0358020, upper bound: 0.0358503
time: 5.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.0919914, -3.0782702, -4.0919914, -3.0782702, -0.3663215, 0.3664084
1: -4.5387135, -3.2085807, -4.5387135, -3.2085807, -0.5401992, 0.5405511
2: -0.5995762, -0.0602678, -0.5995762, -0.0602678, -0.5393084, 0.5393084
3: -1.1253445, -0.7219793, -1.1253445, -0.7219793, -0.1810001, 0.1810045
4: -0.5022298, 0.2107685, -0.5022298, 0.2107685, -0.5793092, 0.5793081
5: -1.6517293, -1.2967341, -1.6517293, -1.2967341, -0.1526992, 0.1528978
6: 0.6691452, 0.7768360, 0.6691452, 0.7768360, -0.0731179, 0.0730057
7: -2.2282453, -1.6877239, -2.2282453, -1.6877239, -0.2644291, 0.2648377
8: -4.9219270, -3.9530594, -4.9219270, -3.9530594, -0.4053394, 0.4053758
9: -4.4394484, -3.4853926, -4.4394484, -3.4853926, -0.3407378, 0.3407769

Time for backsubstitution: 5.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3454
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 921
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 1079
type: DSZ, layer: 1, pos: 1094
type: DSZ, layer: 1, pos: 1109
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3248
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 3503
type: DSZ, layer: 1, pos: 3504

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 576

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0358413, upper bound: 0.0358115
time: 13.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0358481, upper bound: 0.0358009
time: 73.51 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 92.73 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 92.73
Output dim: 3, lower bound: -0.0357947, upper bound: 0.0358517
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 92.73
Output dim: 3, lower bound: -0.0358016, upper bound: 0.0358459
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 92.73
Output dim: 3, lower bound: -0.0358399, upper bound: 0.0358092
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 92.73
Output dim: 3, lower bound: -0.0358505, upper bound: 0.0357987
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 92.73
Output dim: 3, lower bound: -0.0357958, upper bound: 0.0358610
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 92.73
Output dim: 3, lower bound: -0.0358020, upper bound: 0.0358503
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 92.73
Output dim: 3, lower bound: -0.0358413, upper bound: 0.0358115
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 92.73
Output dim: 3, lower bound: -0.0358481, upper bound: 0.0358009

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.0919914, -3.0782702, -4.0919914, -3.0782702, -0.3663830, 0.3662580
1: -4.5387135, -3.2085807, -4.5387135, -3.2085807, -0.5407438, 0.5402568
2: -0.5995762, -0.0602678, -0.5995762, -0.0602678, -0.5393084, 0.5393084
3: -1.1253445, -0.7219793, -1.1253445, -0.7219793, -0.1808838, 0.1809363
4: -0.5022298, 0.2107685, -0.5022298, 0.2107685, -0.5792779, 0.5792780
5: -1.6517293, -1.2967341, -1.6517293, -1.2967341, -0.1529880, 0.1528202
6: 0.6691452, 0.7768360, 0.6691452, 0.7768360, -0.0729102, 0.0730242
7: -2.2282453, -1.6877239, -2.2282453, -1.6877239, -0.2638589, 0.2634283
8: -4.9219270, -3.9530594, -4.9219270, -3.9530594, -0.4052517, 0.4052203
9: -4.4394484, -3.4853926, -4.4394484, -3.4853926, -0.3407103, 0.3405586

Time for backsubstitution: 5.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3454
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 921
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 1079
type: DSZ, layer: 1, pos: 1094
type: DSZ, layer: 1, pos: 1109
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3248
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 3503
type: DSZ, layer: 1, pos: 3504

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2570

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0357647, upper bound: 0.0357153
time: 99.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0356585, upper bound: 0.0358249
time: 7.06 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.0919914, -3.0782702, -4.0919914, -3.0782702, -0.3662759, 0.3663267
1: -4.5387135, -3.2085807, -4.5387135, -3.2085807, -0.5405240, 0.5403414
2: -0.5995762, -0.0602678, -0.5995762, -0.0602678, -0.5393084, 0.5393084
3: -1.1253445, -0.7219793, -1.1253445, -0.7219793, -0.1809074, 0.1808606
4: -0.5022298, 0.2107685, -0.5022298, 0.2107685, -0.5792769, 0.5792791
5: -1.6517293, -1.2967341, -1.6517293, -1.2967341, -0.1530106, 0.1527666
6: 0.6691452, 0.7768360, 0.6691452, 0.7768360, -0.0729120, 0.0730223
7: -2.2282453, -1.6877239, -2.2282453, -1.6877239, -0.2638376, 0.2634499
8: -4.9219270, -3.9530594, -4.9219270, -3.9530594, -0.4052118, 0.4052662
9: -4.4394484, -3.4853926, -4.4394484, -3.4853926, -0.3405510, 0.3406205

Time for backsubstitution: 5.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3454
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 921
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 1079
type: DSZ, layer: 1, pos: 1094
type: DSZ, layer: 1, pos: 1109
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3248
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 3503
type: DSZ, layer: 1, pos: 3504

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2570

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0358173, upper bound: 0.0356622
time: 14.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0357116, upper bound: 0.0357731
time: 6.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.0919914, -3.0782702, -4.0919914, -3.0782702, -0.3663651, 0.3662759
1: -4.5387135, -3.2085807, -4.5387135, -3.2085807, -0.5404766, 0.5405241
2: -0.5995762, -0.0602678, -0.5995762, -0.0602678, -0.5393084, 0.5393084
3: -1.1253445, -0.7219793, -1.1253445, -0.7219793, -0.1808605, 0.1809595
4: -0.5022298, 0.2107685, -0.5022298, 0.2107685, -0.5792791, 0.5792769
5: -1.6517293, -1.2967341, -1.6517293, -1.2967341, -0.1527666, 0.1530416
6: 0.6691452, 0.7768360, 0.6691452, 0.7768360, -0.0730223, 0.0729121
7: -2.2282453, -1.6877239, -2.2282453, -1.6877239, -0.2634499, 0.2638372
8: -4.9219270, -3.9530594, -4.9219270, -3.9530594, -0.4052602, 0.4052118
9: -4.4394484, -3.4853926, -4.4394484, -3.4853926, -0.3407180, 0.3405510

Time for backsubstitution: 5.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3454
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 921
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 1079
type: DSZ, layer: 1, pos: 1094
type: DSZ, layer: 1, pos: 1109
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3248
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 3503
type: DSZ, layer: 1, pos: 3504

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2570

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0357615, upper bound: 0.0357228
time: 4.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0356574, upper bound: 0.0358284
time: 5.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.0919914, -3.0782702, -4.0919914, -3.0782702, -0.3663652, 0.3662758
1: -4.5387135, -3.2085807, -4.5387135, -3.2085807, -0.5404767, 0.5405238
2: -0.5995762, -0.0602678, -0.5995762, -0.0602678, -0.5393084, 0.5393084
3: -1.1253445, -0.7219793, -1.1253445, -0.7219793, -0.1808653, 0.1809547
4: -0.5022298, 0.2107685, -0.5022298, 0.2107685, -0.5792780, 0.5792779
5: -1.6517293, -1.2967341, -1.6517293, -1.2967341, -0.1527665, 0.1530417
6: 0.6691452, 0.7768360, 0.6691452, 0.7768360, -0.0730240, 0.0729105
7: -2.2282453, -1.6877239, -2.2282453, -1.6877239, -0.2634283, 0.2638589
8: -4.9219270, -3.9530594, -4.9219270, -3.9530594, -0.4052593, 0.4052126
9: -4.4394484, -3.4853926, -4.4394484, -3.4853926, -0.3407028, 0.3405660

Time for backsubstitution: 5.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3454
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 921
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 1079
type: DSZ, layer: 1, pos: 1094
type: DSZ, layer: 1, pos: 1109
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3248
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 3503
type: DSZ, layer: 1, pos: 3504

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2570

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0357706, upper bound: 0.0357078
time: 174.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0356650, upper bound: 0.0358231
time: 3.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.0919914, -3.0782702, -4.0919914, -3.0782702, -0.3662580, 0.3663447
1: -4.5387135, -3.2085807, -4.5387135, -3.2085807, -0.5402569, 0.5406086
2: -0.5995762, -0.0602678, -0.5995762, -0.0602678, -0.5393084, 0.5393084
3: -1.1253445, -0.7219793, -1.1253445, -0.7219793, -0.1808842, 0.1808838
4: -0.5022298, 0.2107685, -0.5022298, 0.2107685, -0.5792780, 0.5792780
5: -1.6517293, -1.2967341, -1.6517293, -1.2967341, -0.1527892, 0.1529880
6: 0.6691452, 0.7768360, 0.6691452, 0.7768360, -0.0730241, 0.0729102
7: -2.2282453, -1.6877239, -2.2282453, -1.6877239, -0.2634286, 0.2638589
8: -4.9219270, -3.9530594, -4.9219270, -3.9530594, -0.4052203, 0.4052576
9: -4.4394484, -3.4853926, -4.4394484, -3.4853926, -0.3405586, 0.3406128

Time for backsubstitution: 5.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3454
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 921
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 1079
type: DSZ, layer: 1, pos: 1094
type: DSZ, layer: 1, pos: 1109
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3248
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 3503
type: DSZ, layer: 1, pos: 3504

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2570

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0358152, upper bound: 0.0356690
time: 7.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0357095, upper bound: 0.0357680
time: 112.74 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 126.22 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 126.22
Output dim: 3, lower bound: -0.0357647, upper bound: 0.0357153
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 126.22
Output dim: 3, lower bound: -0.0356585, upper bound: 0.0358249
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 126.22
Output dim: 3, lower bound: -0.0358173, upper bound: 0.0356622
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 126.22
Output dim: 3, lower bound: -0.0357116, upper bound: 0.0357731
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 126.22
Output dim: 3, lower bound: -0.0357615, upper bound: 0.0357228
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 126.22
Output dim: 3, lower bound: -0.0356574, upper bound: 0.0358284
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 126.22
Output dim: 3, lower bound: -0.0357706, upper bound: 0.0357078
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 126.22
Output dim: 3, lower bound: -0.0356650, upper bound: 0.0358231
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 126.22
Output dim: 3, lower bound: -0.0358152, upper bound: 0.0356690
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 126.22
Output dim: 3, lower bound: -0.0357095, upper bound: 0.0357680

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 82.42 + 1714.40 = 1796.82 seconds
