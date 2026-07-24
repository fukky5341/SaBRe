## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 2)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.0253255491


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.3859074, -0.0662569, -0.3859074, -0.0662569, -0.1055312, 0.1055312)
1: (-2.3532712, -1.6709754, -2.3532712, -1.6709754, -0.1567961, 0.1567961)
2: (-1.7074031, -1.2951145, -1.7074031, -1.2951145, -0.1027809, 0.1027809)
3: (-2.5235176, -2.0172057, -2.5235176, -2.0172057, -0.3213125, 0.3213125)
4: (-0.5595140, -0.3382520, -0.5595140, -0.3382520, -0.0941925, 0.0941925)
5: (-2.7547855, -2.1722131, -2.7547855, -2.1722131, -0.3838974, 0.3838974)
6: (-6.0937133, -5.2502670, -6.0937133, -5.2502670, -0.2230141, 0.2230141)
7: (-0.5144494, 0.0605721, -0.5144494, 0.0605721, -0.1401974, 0.1401974)
8: (-0.4112365, -0.0147318, -0.4112365, -0.0147318, -0.1577271, 0.1577271)
9: (0.0308607, 0.3173312, 0.0308607, 0.3173312, -0.0916009, 0.0916009)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.35 + 56.11 = 63.46 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0253505, upper bound: 0.0253500

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 391
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2845
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3145
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3576

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 391

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0251694, upper bound: 0.0253519
time: 18.26 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0253520, upper bound: 0.0251691
time: 166.75 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 185.08 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 185.08
Output dim: 9, lower bound: -0.0251694, upper bound: 0.0253519
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 185.08
Output dim: 9, lower bound: -0.0253520, upper bound: 0.0251691

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.3859074, -0.0662569, -0.3859074, -0.0662569, -0.1055078, 0.1054839
1: -2.3532712, -1.6709754, -2.3532712, -1.6709754, -0.1564737, 0.1563576
2: -1.7074031, -1.2951145, -1.7074031, -1.2951145, -0.1019174, 0.1021445
3: -2.5235176, -2.0172057, -2.5235176, -2.0172057, -0.3211407, 0.3210891
4: -0.5595140, -0.3382520, -0.5595140, -0.3382520, -0.0940174, 0.0940634
5: -2.7547855, -2.1722131, -2.7547855, -2.1722131, -0.3837534, 0.3837273
6: -6.0937133, -5.2502670, -6.0937133, -5.2502670, -0.2224176, 0.2222331
7: -0.5144494, 0.0605721, -0.5144494, 0.0605721, -0.1396284, 0.1397851
8: -0.4112365, -0.0147318, -0.4112365, -0.0147318, -0.1575737, 0.1575127
9: 0.0308607, 0.3173312, 0.0308607, 0.3173312, -0.0912229, 0.0913754

Time for backsubstitution: 6.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2845
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3145
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3576

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 2061

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0251674, upper bound: 0.0253521
time: 7.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0251672, upper bound: 0.0253480
time: 195.28 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.3859074, -0.0662569, -0.3859074, -0.0662569, -0.1054839, 0.1055078
1: -2.3532712, -1.6709754, -2.3532712, -1.6709754, -0.1563576, 0.1564737
2: -1.7074031, -1.2951145, -1.7074031, -1.2951145, -0.1021445, 0.1019174
3: -2.5235176, -2.0172057, -2.5235176, -2.0172057, -0.3210891, 0.3211406
4: -0.5595140, -0.3382520, -0.5595140, -0.3382520, -0.0940634, 0.0940174
5: -2.7547855, -2.1722131, -2.7547855, -2.1722131, -0.3837273, 0.3837534
6: -6.0937133, -5.2502670, -6.0937133, -5.2502670, -0.2222331, 0.2224176
7: -0.5144494, 0.0605721, -0.5144494, 0.0605721, -0.1397851, 0.1396284
8: -0.4112365, -0.0147318, -0.4112365, -0.0147318, -0.1575127, 0.1575737
9: 0.0308607, 0.3173312, 0.0308607, 0.3173312, -0.0913754, 0.0912229

Time for backsubstitution: 6.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2845
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3145
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3576

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2061

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0253470, upper bound: 0.0251720
time: 8.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0253463, upper bound: 0.0251693
time: 78.10 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 92.31 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 92.31
Output dim: 9, lower bound: -0.0251674, upper bound: 0.0253521
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 92.31
Output dim: 9, lower bound: -0.0251672, upper bound: 0.0253480
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 92.31
Output dim: 9, lower bound: -0.0253470, upper bound: 0.0251720
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 92.31
Output dim: 9, lower bound: -0.0253463, upper bound: 0.0251693

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.3859074, -0.0662569, -0.3859074, -0.0662569, -0.1054905, 0.1054667
1: -2.3532712, -1.6709754, -2.3532712, -1.6709754, -0.1559598, 0.1558429
2: -1.7074031, -1.2951145, -1.7074031, -1.2951145, -0.1019018, 0.1021319
3: -2.5235176, -2.0172057, -2.5235176, -2.0172057, -0.3210905, 0.3210469
4: -0.5595140, -0.3382520, -0.5595140, -0.3382520, -0.0940086, 0.0940547
5: -2.7547855, -2.1722131, -2.7547855, -2.1722131, -0.3836946, 0.3836684
6: -6.0937133, -5.2502670, -6.0937133, -5.2502670, -0.2223070, 0.2221311
7: -0.5144494, 0.0605721, -0.5144494, 0.0605721, -0.1396183, 0.1397750
8: -0.4112365, -0.0147318, -0.4112365, -0.0147318, -0.1575717, 0.1575107
9: 0.0308607, 0.3173312, 0.0308607, 0.3173312, -0.0912193, 0.0913722

Time for backsubstitution: 6.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2845
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3145
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3576

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2062

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0251668, upper bound: 0.0253500
time: 7.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0251667, upper bound: 0.0253523
time: 5.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.3859074, -0.0662569, -0.3859074, -0.0662569, -0.1054906, 0.1054667
1: -2.3532712, -1.6709754, -2.3532712, -1.6709754, -0.1559590, 0.1558438
2: -1.7074031, -1.2951145, -1.7074031, -1.2951145, -0.1019049, 0.1021288
3: -2.5235176, -2.0172057, -2.5235176, -2.0172057, -0.3210985, 0.3210389
4: -0.5595140, -0.3382520, -0.5595140, -0.3382520, -0.0940087, 0.0940546
5: -2.7547855, -2.1722131, -2.7547855, -2.1722131, -0.3836945, 0.3836686
6: -6.0937133, -5.2502670, -6.0937133, -5.2502670, -0.2223157, 0.2221225
7: -0.5144494, 0.0605721, -0.5144494, 0.0605721, -0.1396183, 0.1397750
8: -0.4112365, -0.0147318, -0.4112365, -0.0147318, -0.1575717, 0.1575107
9: 0.0308607, 0.3173312, 0.0308607, 0.3173312, -0.0912197, 0.0913719

Time for backsubstitution: 5.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2845
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3145
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3576

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 2062

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0251660, upper bound: 0.0251660
time: 106.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0251666, upper bound: 0.0251665
time: 112.94 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.3859074, -0.0662569, -0.3859074, -0.0662569, -0.1054667, 0.1054906
1: -2.3532712, -1.6709754, -2.3532712, -1.6709754, -0.1558438, 0.1559590
2: -1.7074031, -1.2951145, -1.7074031, -1.2951145, -0.1021288, 0.1019049
3: -2.5235176, -2.0172057, -2.5235176, -2.0172057, -0.3210389, 0.3210985
4: -0.5595140, -0.3382520, -0.5595140, -0.3382520, -0.0940546, 0.0940087
5: -2.7547855, -2.1722131, -2.7547855, -2.1722131, -0.3836685, 0.3836945
6: -6.0937133, -5.2502670, -6.0937133, -5.2502670, -0.2221225, 0.2223157
7: -0.5144494, 0.0605721, -0.5144494, 0.0605721, -0.1397750, 0.1396183
8: -0.4112365, -0.0147318, -0.4112365, -0.0147318, -0.1575107, 0.1575717
9: 0.0308607, 0.3173312, 0.0308607, 0.3173312, -0.0913719, 0.0912197

Time for backsubstitution: 6.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2845
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3145
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3576

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 2062

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0253475, upper bound: 0.0251652
time: 9.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0253447, upper bound: 0.0251643
time: 32.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.3859074, -0.0662569, -0.3859074, -0.0662569, -0.1054667, 0.1054905
1: -2.3532712, -1.6709754, -2.3532712, -1.6709754, -0.1558429, 0.1559598
2: -1.7074031, -1.2951145, -1.7074031, -1.2951145, -0.1021320, 0.1019017
3: -2.5235176, -2.0172057, -2.5235176, -2.0172057, -0.3210469, 0.3210905
4: -0.5595140, -0.3382520, -0.5595140, -0.3382520, -0.0940547, 0.0940086
5: -2.7547855, -2.1722131, -2.7547855, -2.1722131, -0.3836684, 0.3836947
6: -6.0937133, -5.2502670, -6.0937133, -5.2502670, -0.2221311, 0.2223070
7: -0.5144494, 0.0605721, -0.5144494, 0.0605721, -0.1397750, 0.1396183
8: -0.4112365, -0.0147318, -0.4112365, -0.0147318, -0.1575107, 0.1575717
9: 0.0308607, 0.3173312, 0.0308607, 0.3173312, -0.0913722, 0.0912193

Time for backsubstitution: 6.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2845
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3145
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3576

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 2062

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0253455, upper bound: 0.0251618
time: 189.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0253449, upper bound: 0.0251716
time: 9.56 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 205.06 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 205.06
Output dim: 9, lower bound: -0.0251668, upper bound: 0.0253500
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 205.06
Output dim: 9, lower bound: -0.0251667, upper bound: 0.0253523
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 205.06
Output dim: 9, lower bound: -0.0251660, upper bound: 0.0251660
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 205.06
Output dim: 9, lower bound: -0.0251666, upper bound: 0.0251665
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 205.06
Output dim: 9, lower bound: -0.0253475, upper bound: 0.0251652
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 205.06
Output dim: 9, lower bound: -0.0253447, upper bound: 0.0251643
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 205.06
Output dim: 9, lower bound: -0.0253455, upper bound: 0.0251618
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 205.06
Output dim: 9, lower bound: -0.0253449, upper bound: 0.0251716

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.3859074, -0.0662569, -0.3859074, -0.0662569, -0.1054737, 0.1054499
1: -2.3532712, -1.6709754, -2.3532712, -1.6709754, -0.1554488, 0.1553249
2: -1.7074031, -1.2951145, -1.7074031, -1.2951145, -0.1018862, 0.1021182
3: -2.5235176, -2.0172057, -2.5235176, -2.0172057, -0.3210441, 0.3210089
4: -0.5595140, -0.3382520, -0.5595140, -0.3382520, -0.0939982, 0.0940443
5: -2.7547855, -2.1722131, -2.7547855, -2.1722131, -0.3836400, 0.3836135
6: -6.0937133, -5.2502670, -6.0937133, -5.2502670, -0.2221851, 0.2220219
7: -0.5144494, 0.0605721, -0.5144494, 0.0605721, -0.1396099, 0.1397666
8: -0.4112365, -0.0147318, -0.4112365, -0.0147318, -0.1575692, 0.1575083
9: 0.0308607, 0.3173312, 0.0308607, 0.3173312, -0.0912164, 0.0913693

Time for backsubstitution: 6.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2845
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3145
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3576

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 2046

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0251675, upper bound: 0.0253468
time: 75.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0251676, upper bound: 0.0253507
time: 82.36 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.3859074, -0.0662569, -0.3859074, -0.0662569, -0.1054737, 0.1054499
1: -2.3532712, -1.6709754, -2.3532712, -1.6709754, -0.1554418, 0.1553254
2: -1.7074031, -1.2951145, -1.7074031, -1.2951145, -0.1018876, 0.1021164
3: -2.5235176, -2.0172057, -2.5235176, -2.0172057, -0.3210506, 0.3210005
4: -0.5595140, -0.3382520, -0.5595140, -0.3382520, -0.0939982, 0.0940442
5: -2.7547855, -2.1722131, -2.7547855, -2.1722131, -0.3836398, 0.3836137
6: -6.0937133, -5.2502670, -6.0937133, -5.2502670, -0.2221905, 0.2220092
7: -0.5144494, 0.0605721, -0.5144494, 0.0605721, -0.1396099, 0.1397666
8: -0.4112365, -0.0147318, -0.4112365, -0.0147318, -0.1575693, 0.1575083
9: 0.0308607, 0.3173312, 0.0308607, 0.3173312, -0.0912164, 0.0913693

Time for backsubstitution: 6.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2845
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3145
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3576

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 2046

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0251651, upper bound: 0.0253507
time: 8.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0251624, upper bound: 0.0251656
time: 202.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.3859074, -0.0662569, -0.3859074, -0.0662569, -0.1054498, 0.1054737
1: -2.3532712, -1.6709754, -2.3532712, -1.6709754, -0.1553328, 0.1554409
2: -1.7074031, -1.2951145, -1.7074031, -1.2951145, -0.1021132, 0.1018911
3: -2.5235176, -2.0172057, -2.5235176, -2.0172057, -0.3209925, 0.3210604
4: -0.5595140, -0.3382520, -0.5595140, -0.3382520, -0.0940442, 0.0939982
5: -2.7547855, -2.1722131, -2.7547855, -2.1722131, -0.3836139, 0.3836396
6: -6.0937133, -5.2502670, -6.0937133, -5.2502670, -0.2220005, 0.2222065
7: -0.5144494, 0.0605721, -0.5144494, 0.0605721, -0.1397666, 0.1396099
8: -0.4112365, -0.0147318, -0.4112365, -0.0147318, -0.1575083, 0.1575693
9: 0.0308607, 0.3173312, 0.0308607, 0.3173312, -0.0913690, 0.0912168

Time for backsubstitution: 6.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2845
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3145
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3576

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2046

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0253496, upper bound: 0.0251665
time: 183.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0253460, upper bound: 0.0251656
time: 101.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.3859074, -0.0662569, -0.3859074, -0.0662569, -0.1054499, 0.1054737
1: -2.3532712, -1.6709754, -2.3532712, -1.6709754, -0.1553258, 0.1554415
2: -1.7074031, -1.2951145, -1.7074031, -1.2951145, -0.1021147, 0.1018893
3: -2.5235176, -2.0172057, -2.5235176, -2.0172057, -0.3209991, 0.3210521
4: -0.5595140, -0.3382520, -0.5595140, -0.3382520, -0.0940442, 0.0939982
5: -2.7547855, -2.1722131, -2.7547855, -2.1722131, -0.3836136, 0.3836398
6: -6.0937133, -5.2502670, -6.0937133, -5.2502670, -0.2220059, 0.2221937
7: -0.5144494, 0.0605721, -0.5144494, 0.0605721, -0.1397666, 0.1396099
8: -0.4112365, -0.0147318, -0.4112365, -0.0147318, -0.1575083, 0.1575693
9: 0.0308607, 0.3173312, 0.0308607, 0.3173312, -0.0913690, 0.0912167

Time for backsubstitution: 6.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2845
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3145
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3576

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2046

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0253443, upper bound: 0.0251686
time: 9.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0253445, upper bound: 0.0251710
time: 6.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.3859074, -0.0662569, -0.3859074, -0.0662569, -0.1054499, 0.1054737
1: -2.3532712, -1.6709754, -2.3532712, -1.6709754, -0.1553254, 0.1554418
2: -1.7074031, -1.2951145, -1.7074031, -1.2951145, -0.1021164, 0.1018876
3: -2.5235176, -2.0172057, -2.5235176, -2.0172057, -0.3210005, 0.3210506
4: -0.5595140, -0.3382520, -0.5595140, -0.3382520, -0.0940442, 0.0939982
5: -2.7547855, -2.1722131, -2.7547855, -2.1722131, -0.3836136, 0.3836398
6: -6.0937133, -5.2502670, -6.0937133, -5.2502670, -0.2220092, 0.2221905
7: -0.5144494, 0.0605721, -0.5144494, 0.0605721, -0.1397666, 0.1396099
8: -0.4112365, -0.0147318, -0.4112365, -0.0147318, -0.1575083, 0.1575693
9: 0.0308607, 0.3173312, 0.0308607, 0.3173312, -0.0913693, 0.0912164

Time for backsubstitution: 6.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2845
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 3145
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3576

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2046

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0253480, upper bound: 0.0251689
time: 42.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0253467, upper bound: 0.0251703
time: 8.84 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 57.65 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 57.65
Output dim: 9, lower bound: -0.0251675, upper bound: 0.0253468
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 57.65
Output dim: 9, lower bound: -0.0251676, upper bound: 0.0253507
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 57.65
Output dim: 9, lower bound: -0.0251651, upper bound: 0.0253507
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 57.65
Output dim: 9, lower bound: -0.0251624, upper bound: 0.0251656
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 57.65
Output dim: 9, lower bound: -0.0253496, upper bound: 0.0251665
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 57.65
Output dim: 9, lower bound: -0.0253460, upper bound: 0.0251656
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 57.65
Output dim: 9, lower bound: -0.0253443, upper bound: 0.0251686
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 57.65
Output dim: 9, lower bound: -0.0253445, upper bound: 0.0251710
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 57.65
Output dim: 9, lower bound: -0.0253480, upper bound: 0.0251689
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 57.65
Output dim: 9, lower bound: -0.0253467, upper bound: 0.0251703
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 57.65
Output dim: 9, lower bound: -0.0253449, upper bound: 0.0251716

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 63.46 + 1737.10 = 1800.56 seconds
