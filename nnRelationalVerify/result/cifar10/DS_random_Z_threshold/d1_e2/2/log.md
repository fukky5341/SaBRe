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
execution time: IAR + RelationalAnalysis = 7.51 + 54.22 = 61.73 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0253505, upper bound: 0.0253500

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 2845
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 391
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 3145
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 3576
type: DSZ, layer: 1, pos: 700

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2032

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0253506, upper bound: 0.0253508
time: 107.01 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0253474, upper bound: 0.0253468
time: 93.65 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 200.67 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 200.67
Output dim: 9, lower bound: -0.0253506, upper bound: 0.0253508
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 200.67
Output dim: 9, lower bound: -0.0253474, upper bound: 0.0253468

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.3859074, -0.0662569, -0.3859074, -0.0662569, -0.1055312, 0.1055312
1: -2.3532712, -1.6709754, -2.3532712, -1.6709754, -0.1567960, 0.1567960
2: -1.7074031, -1.2951145, -1.7074031, -1.2951145, -0.1027810, 0.1027810
3: -2.5235176, -2.0172057, -2.5235176, -2.0172057, -0.3213129, 0.3213129
4: -0.5595140, -0.3382520, -0.5595140, -0.3382520, -0.0941922, 0.0941922
5: -2.7547855, -2.1722131, -2.7547855, -2.1722131, -0.3838972, 0.3838972
6: -6.0937133, -5.2502670, -6.0937133, -5.2502670, -0.2230132, 0.2230132
7: -0.5144494, 0.0605721, -0.5144494, 0.0605721, -0.1401965, 0.1401966
8: -0.4112365, -0.0147318, -0.4112365, -0.0147318, -0.1577273, 0.1577274
9: 0.0308607, 0.3173312, 0.0308607, 0.3173312, -0.0916008, 0.0916008

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 3576
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 391
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2845
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3145
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2359

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 680

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0253475, upper bound: 0.0253496
time: 67.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0253475, upper bound: 0.0253476
time: 405.46 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.3859074, -0.0662569, -0.3859074, -0.0662569, -0.1055312, 0.1055312
1: -2.3532712, -1.6709754, -2.3532712, -1.6709754, -0.1567960, 0.1567960
2: -1.7074031, -1.2951145, -1.7074031, -1.2951145, -0.1027810, 0.1027810
3: -2.5235176, -2.0172057, -2.5235176, -2.0172057, -0.3213129, 0.3213129
4: -0.5595140, -0.3382520, -0.5595140, -0.3382520, -0.0941922, 0.0941922
5: -2.7547855, -2.1722131, -2.7547855, -2.1722131, -0.3838972, 0.3838972
6: -6.0937133, -5.2502670, -6.0937133, -5.2502670, -0.2230132, 0.2230132
7: -0.5144494, 0.0605721, -0.5144494, 0.0605721, -0.1401965, 0.1401966
8: -0.4112365, -0.0147318, -0.4112365, -0.0147318, -0.1577274, 0.1577274
9: 0.0308607, 0.3173312, 0.0308607, 0.3173312, -0.0916008, 0.0916008

Time for backsubstitution: 6.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 3576
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2845
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 3145
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 391
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2650

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 700

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0253462, upper bound: 0.0253473
time: 435.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0253465, upper bound: 0.0253520
time: 8.91 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 450.74 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 450.74
Output dim: 9, lower bound: -0.0253475, upper bound: 0.0253496
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 450.74
Output dim: 9, lower bound: -0.0253475, upper bound: 0.0253476
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 450.74
Output dim: 9, lower bound: -0.0253462, upper bound: 0.0253473
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 450.74
Output dim: 9, lower bound: -0.0253465, upper bound: 0.0253520

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.3859074, -0.0662569, -0.3859074, -0.0662569, -0.1055311, 0.1055311
1: -2.3532712, -1.6709754, -2.3532712, -1.6709754, -0.1567962, 0.1567962
2: -1.7074031, -1.2951145, -1.7074031, -1.2951145, -0.1027811, 0.1027811
3: -2.5235176, -2.0172057, -2.5235176, -2.0172057, -0.3213130, 0.3213130
4: -0.5595140, -0.3382520, -0.5595140, -0.3382520, -0.0941922, 0.0941922
5: -2.7547855, -2.1722131, -2.7547855, -2.1722131, -0.3838972, 0.3838972
6: -6.0937133, -5.2502670, -6.0937133, -5.2502670, -0.2230133, 0.2230133
7: -0.5144494, 0.0605721, -0.5144494, 0.0605721, -0.1401963, 0.1401964
8: -0.4112365, -0.0147318, -0.4112365, -0.0147318, -0.1577273, 0.1577273
9: 0.0308607, 0.3173312, 0.0308607, 0.3173312, -0.0916008, 0.0916008

Time for backsubstitution: 6.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 3145
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2845
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 391
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 3576
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 710

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 725

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0253472, upper bound: 0.0253537
time: 7.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0253468, upper bound: 0.0253540
time: 5.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.3859074, -0.0662569, -0.3859074, -0.0662569, -0.1055311, 0.1055311
1: -2.3532712, -1.6709754, -2.3532712, -1.6709754, -0.1567962, 0.1567962
2: -1.7074031, -1.2951145, -1.7074031, -1.2951145, -0.1027811, 0.1027811
3: -2.5235176, -2.0172057, -2.5235176, -2.0172057, -0.3213130, 0.3213130
4: -0.5595140, -0.3382520, -0.5595140, -0.3382520, -0.0941922, 0.0941922
5: -2.7547855, -2.1722131, -2.7547855, -2.1722131, -0.3838972, 0.3838972
6: -6.0937133, -5.2502670, -6.0937133, -5.2502670, -0.2230133, 0.2230133
7: -0.5144494, 0.0605721, -0.5144494, 0.0605721, -0.1401963, 0.1401964
8: -0.4112365, -0.0147318, -0.4112365, -0.0147318, -0.1577273, 0.1577273
9: 0.0308607, 0.3173312, 0.0308607, 0.3173312, -0.0916008, 0.0916008

Time for backsubstitution: 6.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 3145
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 391
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 2845
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 3576
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 686

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 724

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0253478, upper bound: 0.0253486
time: 112.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0253471, upper bound: 0.0253532
time: 13.99 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.3859074, -0.0662569, -0.3859074, -0.0662569, -0.1055310, 0.1055310
1: -2.3532712, -1.6709754, -2.3532712, -1.6709754, -0.1567960, 0.1567960
2: -1.7074031, -1.2951145, -1.7074031, -1.2951145, -0.1027810, 0.1027810
3: -2.5235176, -2.0172057, -2.5235176, -2.0172057, -0.3213127, 0.3213127
4: -0.5595140, -0.3382520, -0.5595140, -0.3382520, -0.0941921, 0.0941921
5: -2.7547855, -2.1722131, -2.7547855, -2.1722131, -0.3838970, 0.3838970
6: -6.0937133, -5.2502670, -6.0937133, -5.2502670, -0.2230130, 0.2230129
7: -0.5144494, 0.0605721, -0.5144494, 0.0605721, -0.1401965, 0.1401965
8: -0.4112365, -0.0147318, -0.4112365, -0.0147318, -0.1577267, 0.1577267
9: 0.0308607, 0.3173312, 0.0308607, 0.3173312, -0.0916006, 0.0916006

Time for backsubstitution: 6.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 3145
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 2845
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 391
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 3576
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 3338

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2042

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0253459, upper bound: 0.0253496
time: 83.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0253466, upper bound: 0.0253506
time: 117.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.3859074, -0.0662569, -0.3859074, -0.0662569, -0.1055310, 0.1055310
1: -2.3532712, -1.6709754, -2.3532712, -1.6709754, -0.1567960, 0.1567960
2: -1.7074031, -1.2951145, -1.7074031, -1.2951145, -0.1027810, 0.1027810
3: -2.5235176, -2.0172057, -2.5235176, -2.0172057, -0.3213127, 0.3213127
4: -0.5595140, -0.3382520, -0.5595140, -0.3382520, -0.0941921, 0.0941921
5: -2.7547855, -2.1722131, -2.7547855, -2.1722131, -0.3838970, 0.3838970
6: -6.0937133, -5.2502670, -6.0937133, -5.2502670, -0.2230130, 0.2230129
7: -0.5144494, 0.0605721, -0.5144494, 0.0605721, -0.1401965, 0.1401965
8: -0.4112365, -0.0147318, -0.4112365, -0.0147318, -0.1577267, 0.1577267
9: 0.0308607, 0.3173312, 0.0308607, 0.3173312, -0.0916006, 0.0916006

Time for backsubstitution: 6.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2845
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 391
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 3576
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3145
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 2051

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 818

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0253496, upper bound: 0.0253450
time: 98.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0253452, upper bound: 0.0253540
time: 5.30 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 110.27 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 110.27
Output dim: 9, lower bound: -0.0253472, upper bound: 0.0253537
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 110.27
Output dim: 9, lower bound: -0.0253468, upper bound: 0.0253540
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 110.27
Output dim: 9, lower bound: -0.0253478, upper bound: 0.0253486
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 110.27
Output dim: 9, lower bound: -0.0253471, upper bound: 0.0253532
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 110.27
Output dim: 9, lower bound: -0.0253459, upper bound: 0.0253496
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 110.27
Output dim: 9, lower bound: -0.0253466, upper bound: 0.0253506
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 110.27
Output dim: 9, lower bound: -0.0253496, upper bound: 0.0253450
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 110.27
Output dim: 9, lower bound: -0.0253452, upper bound: 0.0253540

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.3859074, -0.0662569, -0.3859074, -0.0662569, -0.1055195, 0.1055207
1: -2.3532712, -1.6709754, -2.3532712, -1.6709754, -0.1566985, 0.1567055
2: -1.7074031, -1.2951145, -1.7074031, -1.2951145, -0.1027781, 0.1027783
3: -2.5235176, -2.0172057, -2.5235176, -2.0172057, -0.3213022, 0.3213018
4: -0.5595140, -0.3382520, -0.5595140, -0.3382520, -0.0941916, 0.0941916
5: -2.7547855, -2.1722131, -2.7547855, -2.1722131, -0.3838845, 0.3838815
6: -6.0937133, -5.2502670, -6.0937133, -5.2502670, -0.2229666, 0.2229660
7: -0.5144494, 0.0605721, -0.5144494, 0.0605721, -0.1401927, 0.1401927
8: -0.4112365, -0.0147318, -0.4112365, -0.0147318, -0.1577189, 0.1577196
9: 0.0308607, 0.3173312, 0.0308607, 0.3173312, -0.0916001, 0.0916000

Time for backsubstitution: 6.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3145
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2845
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2297
type: DSZ, layer: 1, pos: 3540
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 3142
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2369
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 2171
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 886
type: DSZ, layer: 1, pos: 712
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 887
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 2070
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2391
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 3122
type: DSZ, layer: 1, pos: 3576
type: DSZ, layer: 1, pos: 391
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 687
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 615
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 693
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2359

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2045

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0253493, upper bound: 0.0253492
time: 162.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0253476, upper bound: 0.0253497
time: 8.06 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 176.60 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 176.60
Output dim: 9, lower bound: -0.0253493, upper bound: 0.0253492
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 176.60
Output dim: 9, lower bound: -0.0253476, upper bound: 0.0253497
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 176.60
Output dim: 9, lower bound: -0.0253468, upper bound: 0.0253540
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 176.60
Output dim: 9, lower bound: -0.0253478, upper bound: 0.0253486
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 176.60
Output dim: 9, lower bound: -0.0253471, upper bound: 0.0253532
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 176.60
Output dim: 9, lower bound: -0.0253459, upper bound: 0.0253496
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 176.60
Output dim: 9, lower bound: -0.0253466, upper bound: 0.0253506
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 176.60
Output dim: 9, lower bound: -0.0253496, upper bound: 0.0253450
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 176.60
Output dim: 9, lower bound: -0.0253452, upper bound: 0.0253540

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 61.73 + 1776.58 = 1838.32 seconds
