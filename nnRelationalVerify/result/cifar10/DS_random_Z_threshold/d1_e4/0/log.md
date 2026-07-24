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
execution time: IAR + RelationalAnalysis = 8.12 + 75.95 = 84.08 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0359040, upper bound: 0.0359104

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 1109
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 921
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 1079
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 322
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 3248
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 1094
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 3454
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 3503
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2404

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2562

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0358933, upper bound: 0.0358867
time: 75.57 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0358817, upper bound: 0.0358856
time: 457.86 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 533.45 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 533.45
Output dim: 3, lower bound: -0.0358933, upper bound: 0.0358867
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 533.45
Output dim: 3, lower bound: -0.0358817, upper bound: 0.0358856

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -4.0919914, -3.0782702, -4.0919914, -3.0782702, -0.3675562, 0.3676271
1: -4.5387135, -3.2085807, -4.5387135, -3.2085807, -0.5487077, 0.5487518
2: -0.5995762, -0.0602678, -0.5995762, -0.0602678, -0.5393084, 0.5393084
3: -1.1253445, -0.7219793, -1.1253445, -0.7219793, -0.1811064, 0.1810912
4: -0.5022298, 0.2107685, -0.5022298, 0.2107685, -0.5792912, 0.5792953
5: -1.6517293, -1.2967341, -1.6517293, -1.2967341, -0.1554690, 0.1554503
6: 0.6691452, 0.7768360, 0.6691452, 0.7768360, -0.0744236, 0.0744236
7: -2.2282453, -1.6877239, -2.2282453, -1.6877239, -0.2697674, 0.2697660
8: -4.9219270, -3.9530594, -4.9219270, -3.9530594, -0.4045534, 0.4046153
9: -4.4394484, -3.4853926, -4.4394484, -3.4853926, -0.3406574, 0.3406777

Time for backsubstitution: 6.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 3503
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 322
type: DSZ, layer: 1, pos: 921
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 3248
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 1079
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 3454
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 1094
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 1109
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 3143

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 708

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0358890, upper bound: 0.0358891
time: 115.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0358925, upper bound: 0.0358804
time: 237.34 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -4.0919914, -3.0782702, -4.0919914, -3.0782702, -0.3676271, 0.3675562
1: -4.5387135, -3.2085807, -4.5387135, -3.2085807, -0.5487518, 0.5487077
2: -0.5995762, -0.0602678, -0.5995762, -0.0602678, -0.5393084, 0.5393084
3: -1.1253445, -0.7219793, -1.1253445, -0.7219793, -0.1810912, 0.1811064
4: -0.5022298, 0.2107685, -0.5022298, 0.2107685, -0.5792953, 0.5792913
5: -1.6517293, -1.2967341, -1.6517293, -1.2967341, -0.1554503, 0.1554690
6: 0.6691452, 0.7768360, 0.6691452, 0.7768360, -0.0744236, 0.0744236
7: -2.2282453, -1.6877239, -2.2282453, -1.6877239, -0.2697660, 0.2697674
8: -4.9219270, -3.9530594, -4.9219270, -3.9530594, -0.4046152, 0.4045534
9: -4.4394484, -3.4853926, -4.4394484, -3.4853926, -0.3406777, 0.3406575

Time for backsubstitution: 6.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 3503
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 322
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3454
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 921
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 1094
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3248
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 1109
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 1079
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 2640

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 816

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0358555, upper bound: 0.0358838
time: 80.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0358671, upper bound: 0.0358755
time: 6.32 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 92.85 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 92.85
Output dim: 3, lower bound: -0.0358890, upper bound: 0.0358891
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 92.85
Output dim: 3, lower bound: -0.0358925, upper bound: 0.0358804
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 92.85
Output dim: 3, lower bound: -0.0358555, upper bound: 0.0358838
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 92.85
Output dim: 3, lower bound: -0.0358671, upper bound: 0.0358755

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.0919914, -3.0782702, -4.0919914, -3.0782702, -0.3671746, 0.3672366
1: -4.5387135, -3.2085807, -4.5387135, -3.2085807, -0.5486668, 0.5487106
2: -0.5995762, -0.0602678, -0.5995762, -0.0602678, -0.5393084, 0.5393084
3: -1.1253445, -0.7219793, -1.1253445, -0.7219793, -0.1810466, 0.1810337
4: -0.5022298, 0.2107685, -0.5022298, 0.2107685, -0.5792333, 0.5792373
5: -1.6517293, -1.2967341, -1.6517293, -1.2967341, -0.1553833, 0.1553649
6: 0.6691452, 0.7768360, 0.6691452, 0.7768360, -0.0744106, 0.0744109
7: -2.2282453, -1.6877239, -2.2282453, -1.6877239, -0.2697448, 0.2697432
8: -4.9219270, -3.9530594, -4.9219270, -3.9530594, -0.4044499, 0.4044999
9: -4.4394484, -3.4853926, -4.4394484, -3.4853926, -0.3406573, 0.3406775

Time for backsubstitution: 6.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 1079
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 322
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 1094
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 3503
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 3454
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 1109
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 3248
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 921
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2631

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 908

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0358907, upper bound: 0.0358871
time: 34.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0358895, upper bound: 0.0358885
time: 32.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.0919914, -3.0782702, -4.0919914, -3.0782702, -0.3671656, 0.3672456
1: -4.5387135, -3.2085807, -4.5387135, -3.2085807, -0.5486665, 0.5487108
2: -0.5995762, -0.0602678, -0.5995762, -0.0602678, -0.5393084, 0.5393084
3: -1.1253445, -0.7219793, -1.1253445, -0.7219793, -0.1810488, 0.1810315
4: -0.5022298, 0.2107685, -0.5022298, 0.2107685, -0.5792333, 0.5792373
5: -1.6517293, -1.2967341, -1.6517293, -1.2967341, -0.1553837, 0.1553646
6: 0.6691452, 0.7768360, 0.6691452, 0.7768360, -0.0744108, 0.0744106
7: -2.2282453, -1.6877239, -2.2282453, -1.6877239, -0.2697446, 0.2697433
8: -4.9219270, -3.9530594, -4.9219270, -3.9530594, -0.4044381, 0.4045118
9: -4.4394484, -3.4853926, -4.4394484, -3.4853926, -0.3406572, 0.3406775

Time for backsubstitution: 6.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 322
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 1094
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 1079
type: DSZ, layer: 1, pos: 3248
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 921
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 1109
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 3503
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 3454
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 2569

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0358912, upper bound: 0.0358799
time: 116.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0358894, upper bound: 0.0358856
time: 6.13 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.0919914, -3.0782702, -4.0919914, -3.0782702, -0.3656178, 0.3655243
1: -4.5387135, -3.2085807, -4.5387135, -3.2085807, -0.5462508, 0.5461792
2: -0.5995762, -0.0602678, -0.5995762, -0.0602678, -0.5393084, 0.5393084
3: -1.1253445, -0.7219793, -1.1253445, -0.7219793, -0.1804972, 0.1805177
4: -0.5022298, 0.2107685, -0.5022298, 0.2107685, -0.5792924, 0.5792884
5: -1.6517293, -1.2967341, -1.6517293, -1.2967341, -0.1547453, 0.1547692
6: 0.6691452, 0.7768360, 0.6691452, 0.7768360, -0.0744213, 0.0744214
7: -2.2282453, -1.6877239, -2.2282453, -1.6877239, -0.2697615, 0.2697630
8: -4.9219270, -3.9530594, -4.9219270, -3.9530594, -0.4034916, 0.4034158
9: -4.4394484, -3.4853926, -4.4394484, -3.4853926, -0.3397240, 0.3396955

Time for backsubstitution: 6.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 3503
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 1109
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 1079
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 322
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 3248
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3454
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 1094
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 921
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2536

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 677

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0358582, upper bound: 0.0358834
time: 183.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0358598, upper bound: 0.0358881
time: 5.77 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.0919914, -3.0782702, -4.0919914, -3.0782702, -0.3655953, 0.3655468
1: -4.5387135, -3.2085807, -4.5387135, -3.2085807, -0.5462233, 0.5462068
2: -0.5995762, -0.0602678, -0.5995762, -0.0602678, -0.5393084, 0.5393084
3: -1.1253445, -0.7219793, -1.1253445, -0.7219793, -0.1805025, 0.1805124
4: -0.5022298, 0.2107685, -0.5022298, 0.2107685, -0.5792924, 0.5792884
5: -1.6517293, -1.2967341, -1.6517293, -1.2967341, -0.1547504, 0.1547640
6: 0.6691452, 0.7768360, 0.6691452, 0.7768360, -0.0744214, 0.0744213
7: -2.2282453, -1.6877239, -2.2282453, -1.6877239, -0.2697616, 0.2697629
8: -4.9219270, -3.9530594, -4.9219270, -3.9530594, -0.4034776, 0.4034297
9: -4.4394484, -3.4853926, -4.4394484, -3.4853926, -0.3397158, 0.3397037

Time for backsubstitution: 6.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 3503
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 1109
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 3454
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 1094
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 322
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 1079
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 921
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 3248
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 338

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2107

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0358314, upper bound: 0.0358573
time: 129.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0358530, upper bound: 0.0358392
time: 96.51 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 232.34 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 232.34
Output dim: 3, lower bound: -0.0358907, upper bound: 0.0358871
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 232.34
Output dim: 3, lower bound: -0.0358895, upper bound: 0.0358885
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 232.34
Output dim: 3, lower bound: -0.0358912, upper bound: 0.0358799
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 232.34
Output dim: 3, lower bound: -0.0358894, upper bound: 0.0358856
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 232.34
Output dim: 3, lower bound: -0.0358582, upper bound: 0.0358834
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 232.34
Output dim: 3, lower bound: -0.0358598, upper bound: 0.0358881
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 232.34
Output dim: 3, lower bound: -0.0358314, upper bound: 0.0358573
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 232.34
Output dim: 3, lower bound: -0.0358530, upper bound: 0.0358392

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.0919914, -3.0782702, -4.0919914, -3.0782702, -0.3671745, 0.3672365
1: -4.5387135, -3.2085807, -4.5387135, -3.2085807, -0.5486668, 0.5487106
2: -0.5995762, -0.0602678, -0.5995762, -0.0602678, -0.5393084, 0.5393084
3: -1.1253445, -0.7219793, -1.1253445, -0.7219793, -0.1810466, 0.1810337
4: -0.5022298, 0.2107685, -0.5022298, 0.2107685, -0.5792333, 0.5792373
5: -1.6517293, -1.2967341, -1.6517293, -1.2967341, -0.1553833, 0.1553649
6: 0.6691452, 0.7768360, 0.6691452, 0.7768360, -0.0744105, 0.0744108
7: -2.2282453, -1.6877239, -2.2282453, -1.6877239, -0.2697448, 0.2697432
8: -4.9219270, -3.9530594, -4.9219270, -3.9530594, -0.4044499, 0.4044999
9: -4.4394484, -3.4853926, -4.4394484, -3.4853926, -0.3406571, 0.3406774

Time for backsubstitution: 6.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 1109
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 2502
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 1079
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 1094
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 3454
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 2660
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 2108
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 3503
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 2640
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2578
type: DSZ, layer: 1, pos: 923
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 2143
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 322
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 921
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 3059
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3248
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 2099

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2107

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0358524, upper bound: 0.0358726
time: 114.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0358738, upper bound: 0.0358471
time: 176.61 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 297.32 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 297.32
Output dim: 3, lower bound: -0.0358524, upper bound: 0.0358726
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 297.32
Output dim: 3, lower bound: -0.0358738, upper bound: 0.0358471
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 297.32
Output dim: 3, lower bound: -0.0358895, upper bound: 0.0358885
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 297.32
Output dim: 3, lower bound: -0.0358912, upper bound: 0.0358799
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 297.32
Output dim: 3, lower bound: -0.0358894, upper bound: 0.0358856
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 297.32
Output dim: 3, lower bound: -0.0358582, upper bound: 0.0358834
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 297.32
Output dim: 3, lower bound: -0.0358598, upper bound: 0.0358881
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 297.32
Output dim: 3, lower bound: -0.0358314, upper bound: 0.0358573
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 297.32
Output dim: 3, lower bound: -0.0358530, upper bound: 0.0358392

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 84.08 + 1913.01 = 1997.08 seconds
