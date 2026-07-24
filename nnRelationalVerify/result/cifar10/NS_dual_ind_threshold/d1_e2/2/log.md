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
execution time: IAR + RelationalAnalysis = 7.14 + 54.80 = 61.94 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0253505, upper bound: 0.0253500

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3566
type: A, layer: 1, pos: 3556
type: A, layer: 1, pos: 391
type: A, layer: 1, pos: 3562
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 3553
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 408
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2434
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 3300
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 3051
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 2086
type: A, layer: 1, pos: 3017
type: A, layer: 1, pos: 2359
type: A, layer: 1, pos: 2411
type: A, layer: 1, pos: 3576
type: A, layer: 1, pos: 2070
type: A, layer: 1, pos: 3540
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 3041
type: A, layer: 1, pos: 2391
type: A, layer: 1, pos: 3476
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 3034
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 2972
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2297
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 2845
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2395
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 3338
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2028
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 3008
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2029
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2046
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 3122
type: A, layer: 1, pos: 3335
type: A, layer: 1, pos: 3267
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2045
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 2027
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 2037
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 149
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2369
type: A, layer: 1, pos: 2693
type: A, layer: 1, pos: 2694
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 2999
type: A, layer: 1, pos: 3142
type: A, layer: 1, pos: 3145
type: A, layer: 1, pos: 3554

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3566

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0253458, upper bound: 0.0251762
time: 7.07 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0253451, upper bound: 0.0253515
time: 5.40 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 12.54 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 12.54
Output dim: 9, lower bound: -0.0253458, upper bound: 0.0251762
NS_A2, status: Status.UNKNOWN, split count: 1, time: 12.54
Output dim: 9, lower bound: -0.0253451, upper bound: 0.0253515

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.3851550, -0.0666637, -0.3853086, -0.0662591, -0.1047427, 0.1044619
1: -2.3523502, -1.6720414, -2.3532383, -1.6718485, -0.1545273, 0.1555699
2: -1.7073590, -1.2955428, -1.7073679, -1.2951312, -0.1027227, 0.1022500
3: -2.5229864, -2.0181522, -2.5234571, -2.0179868, -0.3198521, 0.3202406
4: -0.5593612, -0.3383261, -0.5594731, -0.3382556, -0.0939977, 0.0940430
5: -2.7540708, -2.1734195, -2.7546971, -2.1731925, -0.3821744, 0.3826121
6: -6.0931067, -5.2517281, -6.0936904, -5.2513371, -0.2210778, 0.2215972
7: -0.5140445, 0.0604134, -0.5141121, 0.0605685, -0.1397662, 0.1395833
8: -0.4108537, -0.0150289, -0.4109188, -0.0148998, -0.1573160, 0.1568708
9: 0.0324989, 0.3161106, 0.0308970, 0.3163405, -0.0887751, 0.0902431

Time for backsubstitution: 5.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3556
type: B, layer: 1, pos: 391
type: B, layer: 1, pos: 3562
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 3553
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 408
type: B, layer: 1, pos: 3566
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2434
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 3081
type: B, layer: 1, pos: 3300
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 3051
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2086
type: B, layer: 1, pos: 3017
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 2411
type: B, layer: 1, pos: 3576
type: B, layer: 1, pos: 2070
type: B, layer: 1, pos: 3540
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 3041
type: B, layer: 1, pos: 2391
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 3034
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 2972
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2297
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 2845
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2395
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 3338
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2028
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 3008
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2029
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2046
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 3335
type: B, layer: 1, pos: 3267
type: B, layer: 1, pos: 3122
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 2030
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2045
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 2027
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 2037
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 2036
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 3462
type: B, layer: 1, pos: 2051
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 149
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2069
type: B, layer: 1, pos: 2369
type: B, layer: 1, pos: 2693
type: B, layer: 1, pos: 2694
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 2999
type: B, layer: 1, pos: 3142
type: B, layer: 1, pos: 3145
type: B, layer: 1, pos: 3554

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3556

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0251887, upper bound: 0.0251716
time: 139.13 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0253466, upper bound: 0.0251709
time: 65.11 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.3858984, -0.0662569, -0.3859002, -0.0662569, -0.1045352, 0.1055306
1: -2.3532708, -1.6710639, -2.3532710, -1.6710458, -0.1567945, 0.1549883
2: -1.7073991, -1.2951144, -1.7074001, -1.2951144, -0.1027182, 0.1027698
3: -2.5235164, -2.0172508, -2.5235167, -2.0172415, -0.3213070, 0.3206041
4: -0.5595141, -0.3382521, -0.5595140, -0.3382521, -0.0941696, 0.0941769
5: -2.7547846, -2.1722322, -2.7547841, -2.1722286, -0.3838910, 0.3833589
6: -6.0937119, -5.2503691, -6.0937123, -5.2503486, -0.2229837, 0.2212340
7: -0.5144150, 0.0605721, -0.5144221, 0.0605721, -0.1395541, 0.1401754
8: -0.4112353, -0.0147318, -0.4112356, -0.0147317, -0.1575848, 0.1580705
9: 0.0308607, 0.3173208, 0.0308606, 0.3173226, -0.0916007, 0.0896917

Time for backsubstitution: 5.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3556
type: B, layer: 1, pos: 391
type: B, layer: 1, pos: 3562
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 3553
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 408
type: B, layer: 1, pos: 3566
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2434
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 3081
type: B, layer: 1, pos: 3300
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 3051
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2086
type: B, layer: 1, pos: 3017
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 2411
type: B, layer: 1, pos: 3576
type: B, layer: 1, pos: 2070
type: B, layer: 1, pos: 3540
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 3041
type: B, layer: 1, pos: 2391
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 3034
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 2972
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2297
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 2845
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2395
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 3338
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2028
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 3008
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2029
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2046
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 3122
type: B, layer: 1, pos: 3335
type: B, layer: 1, pos: 3267
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 2030
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2045
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 2027
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 2037
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 2036
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 3462
type: B, layer: 1, pos: 2051
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 149
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2069
type: B, layer: 1, pos: 2369
type: B, layer: 1, pos: 2693
type: B, layer: 1, pos: 2694
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 2999
type: B, layer: 1, pos: 3142
type: B, layer: 1, pos: 3145
type: B, layer: 1, pos: 3554

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3556

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0251906, upper bound: 0.0253475
time: 172.92 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0253445, upper bound: 0.0251759
time: 84.18 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 262.64 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 262.64
Output dim: 9, lower bound: -0.0251887, upper bound: 0.0251716
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 262.64
Output dim: 9, lower bound: -0.0253466, upper bound: 0.0251709
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 262.64
Output dim: 9, lower bound: -0.0251906, upper bound: 0.0253475
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 262.64
Output dim: 9, lower bound: -0.0253445, upper bound: 0.0251759

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.3851452, -0.0666638, -0.3852963, -0.0662592, -0.1047364, 0.1031708
1: -2.3523498, -1.6720948, -2.3532386, -1.6719170, -0.1530182, 0.1555561
2: -1.7073506, -1.2955428, -1.7073578, -1.2951313, -0.1026981, 0.1022251
3: -2.5229840, -2.0181625, -2.5234537, -2.0179992, -0.3184056, 0.3202276
4: -0.5593492, -0.3383261, -0.5594590, -0.3382556, -0.0939914, 0.0939924
5: -2.7540689, -2.1734223, -2.7546942, -2.1731961, -0.3816411, 0.3826025
6: -6.0931067, -5.2517405, -6.0936899, -5.2513523, -0.2187165, 0.2215739
7: -0.5140445, 0.0604023, -0.5141121, 0.0605542, -0.1396063, 0.1395829
8: -0.4108299, -0.0150292, -0.4108882, -0.0149002, -0.1573149, 0.1568424
9: 0.0324990, 0.3161068, 0.0308972, 0.3163357, -0.0873227, 0.0902398

Time for backsubstitution: 5.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 391
type: A, layer: 1, pos: 3562
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 3553
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 408
type: A, layer: 1, pos: 3556
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2434
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 3300
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 3051
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 2086
type: A, layer: 1, pos: 3017
type: A, layer: 1, pos: 2359
type: A, layer: 1, pos: 2411
type: A, layer: 1, pos: 3576
type: A, layer: 1, pos: 2070
type: A, layer: 1, pos: 3540
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 3041
type: A, layer: 1, pos: 2391
type: A, layer: 1, pos: 3476
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 3034
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 2972
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2297
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 2845
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2395
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 3338
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2028
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 3008
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2029
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2046
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 3122
type: A, layer: 1, pos: 3335
type: A, layer: 1, pos: 3267
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2045
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 2027
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 2037
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 149
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2369
type: A, layer: 1, pos: 2693
type: A, layer: 1, pos: 2694
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 2999
type: A, layer: 1, pos: 3142
type: A, layer: 1, pos: 3145
type: A, layer: 1, pos: 3554

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 391

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0251607, upper bound: 0.0251761
time: 6.53 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0253445, upper bound: 0.0251754
time: 110.15 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.3851480, -0.0662584, -0.3849765, -0.0666087, -0.1033565, 0.1045780
1: -2.3532476, -1.6719551, -2.3526011, -1.6721575, -0.1557042, 0.1532484
2: -1.7073754, -1.2951267, -1.7073685, -1.2952745, -0.1024419, 0.1027360
3: -2.5234687, -2.0185027, -2.5227745, -2.0187936, -0.3198252, 0.3187551
4: -0.5594840, -0.3382547, -0.5594753, -0.3382536, -0.0940993, 0.0941306
5: -2.7547340, -2.1730630, -2.7542725, -2.1732671, -0.3828784, 0.3820941
6: -6.0937104, -5.2518001, -6.0929446, -5.2521214, -0.2212382, 0.2190571
7: -0.5144140, 0.0605031, -0.5144098, 0.0604846, -0.1394362, 0.1400275
8: -0.4110895, -0.0147844, -0.4110522, -0.0148278, -0.1572736, 0.1578186
9: 0.0308859, 0.3164655, 0.0319584, 0.3162527, -0.0905255, 0.0877419

Time for backsubstitution: 5.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 391
type: A, layer: 1, pos: 3562
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 3553
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 408
type: A, layer: 1, pos: 3556
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2434
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 3300
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 3051
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 2086
type: A, layer: 1, pos: 3017
type: A, layer: 1, pos: 2359
type: A, layer: 1, pos: 2411
type: A, layer: 1, pos: 3576
type: A, layer: 1, pos: 2070
type: A, layer: 1, pos: 3540
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 3041
type: A, layer: 1, pos: 2391
type: A, layer: 1, pos: 3476
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 3034
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 2972
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2297
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 2845
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2395
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 3338
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2028
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 3008
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2029
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2046
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 3335
type: A, layer: 1, pos: 3267
type: A, layer: 1, pos: 3122
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2045
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 2027
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 2037
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 149
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2369
type: A, layer: 1, pos: 2693
type: A, layer: 1, pos: 2694
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 2999
type: A, layer: 1, pos: 3142
type: A, layer: 1, pos: 3145
type: A, layer: 1, pos: 3554

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 391

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0250159, upper bound: 0.0253507
time: 138.84 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0251901, upper bound: 0.0253497
time: 10.85 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.3858886, -0.0662569, -0.3858877, -0.0662568, -0.1045288, 0.1042395
1: -2.3532703, -1.6711168, -2.3532708, -1.6711140, -0.1552853, 0.1549746
2: -1.7073913, -1.2951146, -1.7073898, -1.2951145, -0.1026937, 0.1027449
3: -2.5235140, -2.0172610, -2.5235131, -2.0172544, -0.3198605, 0.3205913
4: -0.5595022, -0.3382521, -0.5594999, -0.3382521, -0.0941633, 0.0941263
5: -2.7547822, -2.1722350, -2.7547815, -2.1722317, -0.3833576, 0.3833490
6: -6.0937119, -5.2503805, -6.0937119, -5.2503633, -0.2206225, 0.2212107
7: -0.5144149, 0.0605609, -0.5144219, 0.0605577, -0.1393942, 0.1401751
8: -0.4112115, -0.0147319, -0.4112051, -0.0147320, -0.1575837, 0.1580422
9: 0.0308607, 0.3173170, 0.0308607, 0.3173178, -0.0901482, 0.0896883

Time for backsubstitution: 5.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 391
type: A, layer: 1, pos: 3562
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 3553
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 408
type: A, layer: 1, pos: 3556
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2434
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 3300
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 3051
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 2086
type: A, layer: 1, pos: 3017
type: A, layer: 1, pos: 2359
type: A, layer: 1, pos: 2411
type: A, layer: 1, pos: 3576
type: A, layer: 1, pos: 2070
type: A, layer: 1, pos: 3540
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 3041
type: A, layer: 1, pos: 2391
type: A, layer: 1, pos: 3476
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 3034
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 2972
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2297
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 2845
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2395
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 3338
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2028
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 3008
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2029
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2046
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 3122
type: A, layer: 1, pos: 3335
type: A, layer: 1, pos: 3267
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 2045
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 2027
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 2037
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 2036
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 149
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2054
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2369
type: A, layer: 1, pos: 2693
type: A, layer: 1, pos: 2694
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 2999
type: A, layer: 1, pos: 3142
type: A, layer: 1, pos: 3145
type: A, layer: 1, pos: 3554

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 391

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0251610, upper bound: 0.0253520
time: 39.48 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0253467, upper bound: 0.0253454
time: 188.64 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 233.63 seconds
NS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 233.63
Output dim: 9, lower bound: -0.0251607, upper bound: 0.0251761
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 233.63
Output dim: 9, lower bound: -0.0253445, upper bound: 0.0251754
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 233.63
Output dim: 9, lower bound: -0.0250159, upper bound: 0.0253507
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 233.63
Output dim: 9, lower bound: -0.0251901, upper bound: 0.0253497
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 233.63
Output dim: 9, lower bound: -0.0251610, upper bound: 0.0253520
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 233.63
Output dim: 9, lower bound: -0.0253467, upper bound: 0.0253454

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.3853895, -0.0666920, -0.3852957, -0.0662806, -0.1047088, 0.1034948
1: -2.3549757, -1.6720779, -2.3532312, -1.6719441, -0.1556059, 0.1552190
2: -1.7073681, -1.2908833, -1.7073197, -1.2951376, -0.1020644, 0.1068165
3: -2.5240083, -2.0182822, -2.5234485, -2.0181198, -0.3194844, 0.3200917
4: -0.5593922, -0.3368742, -0.5594323, -0.3382562, -0.0938590, 0.0954272
5: -2.7548385, -2.1735065, -2.7546878, -2.1732874, -0.3824086, 0.3826558
6: -6.0968924, -5.2520666, -6.0936861, -5.2516956, -0.2228540, 0.2209174
7: -0.5140707, 0.0633765, -0.5141050, 0.0605541, -0.1391872, 0.1425012
8: -0.4126561, -0.0147838, -0.4108879, -0.0149033, -0.1591197, 0.1566938
9: 0.0325041, 0.3184936, 0.0309128, 0.3163355, -0.0872062, 0.0923413

Time for backsubstitution: 5.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3562
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 3553
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 408
type: B, layer: 1, pos: 3566
type: B, layer: 1, pos: 391
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2434
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 3081
type: B, layer: 1, pos: 3300
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 3051
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2086
type: B, layer: 1, pos: 3017
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 2411
type: B, layer: 1, pos: 3576
type: B, layer: 1, pos: 2070
type: B, layer: 1, pos: 3540
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 3041
type: B, layer: 1, pos: 2391
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 3034
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 2972
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2297
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 2845
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2395
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 3338
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2028
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 3008
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2029
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2046
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 3335
type: B, layer: 1, pos: 3267
type: B, layer: 1, pos: 3122
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 2030
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2045
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 2027
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 2037
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 2036
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 3462
type: B, layer: 1, pos: 2051
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 149
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2069
type: B, layer: 1, pos: 2369
type: B, layer: 1, pos: 2693
type: B, layer: 1, pos: 2694
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 2999
type: B, layer: 1, pos: 3142
type: B, layer: 1, pos: 3145
type: B, layer: 1, pos: 3554

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3562

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0252818, upper bound: 0.0251657
time: 94.81 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0253390, upper bound: 0.0251630
time: 141.37 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.3851250, -0.0662694, -0.3849604, -0.0666161, -0.1032340, 0.1044715
1: -2.3532286, -1.6725430, -2.3525887, -1.6725628, -0.1552505, 0.1526103
2: -1.7063121, -1.2951424, -1.7066236, -1.2952850, -0.1013489, 0.1019671
3: -2.5234418, -2.0187204, -2.5227551, -2.0189466, -0.3196312, 0.3184963
4: -0.5591420, -0.3382562, -0.5592365, -0.3382545, -0.0937444, 0.0938817
5: -2.7546887, -2.1731830, -2.7542408, -2.1733551, -0.3827001, 0.3818752
6: -6.0937047, -5.2526240, -6.0929408, -5.2526994, -0.2205658, 0.2181368
7: -0.5137150, 0.0605009, -0.5139284, 0.0604832, -0.1387560, 0.1395440
8: -0.4110884, -0.0151454, -0.4110516, -0.0150763, -0.1570038, 0.1574332
9: 0.0314446, 0.3164591, 0.0323430, 0.3162485, -0.0900708, 0.0873947

Time for backsubstitution: 5.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3562
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 3553
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 408
type: B, layer: 1, pos: 3566
type: B, layer: 1, pos: 391
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2434
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 3081
type: B, layer: 1, pos: 3300
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 3051
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2086
type: B, layer: 1, pos: 3017
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 2411
type: B, layer: 1, pos: 3576
type: B, layer: 1, pos: 2070
type: B, layer: 1, pos: 3540
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 3041
type: B, layer: 1, pos: 2391
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 3034
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 2972
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2297
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 2845
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2395
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 3338
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2028
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 3008
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2029
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2046
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 3122
type: B, layer: 1, pos: 3335
type: B, layer: 1, pos: 3267
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 2030
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2045
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 2027
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 2037
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 2036
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 3462
type: B, layer: 1, pos: 2051
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 149
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2069
type: B, layer: 1, pos: 2369
type: B, layer: 1, pos: 2693
type: B, layer: 1, pos: 2694
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 2999
type: B, layer: 1, pos: 3142
type: B, layer: 1, pos: 3145
type: B, layer: 1, pos: 3554

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3562

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0249501, upper bound: 0.0253416
time: 11.24 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0250090, upper bound: 0.0253381
time: 9.46 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.3853884, -0.0662866, -0.3849760, -0.0666310, -0.1033136, 0.1048933
1: -2.3558729, -1.6719389, -2.3525941, -1.6721818, -0.1582075, 0.1529120
2: -1.7073917, -1.2904674, -1.7073306, -1.2952807, -0.1018066, 0.1073248
3: -2.5244927, -2.0186090, -2.5227690, -2.0188994, -0.3208960, 0.3186121
4: -0.5595270, -0.3368028, -0.5594493, -0.3382542, -0.0939661, 0.0955654
5: -2.7555032, -2.1731472, -2.7542660, -2.1733587, -0.3836580, 0.3821390
6: -6.0974970, -5.2520638, -6.0929399, -5.2524171, -0.2252492, 0.2183903
7: -0.5144392, 0.0634773, -0.5144031, 0.0604846, -0.1390171, 0.1429461
8: -0.4129149, -0.0145402, -0.4110519, -0.0148309, -0.1590774, 0.1576698
9: 0.0308911, 0.3188500, 0.0319742, 0.3162526, -0.0904061, 0.0898385

Time for backsubstitution: 5.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3562
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 3553
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 408
type: B, layer: 1, pos: 3566
type: B, layer: 1, pos: 391
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2434
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 3081
type: B, layer: 1, pos: 3300
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 3051
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2086
type: B, layer: 1, pos: 3017
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 2411
type: B, layer: 1, pos: 3576
type: B, layer: 1, pos: 2070
type: B, layer: 1, pos: 3540
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 3041
type: B, layer: 1, pos: 2391
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 3034
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 2972
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2297
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 2845
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2395
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 3338
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2028
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 3008
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2029
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2046
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 3122
type: B, layer: 1, pos: 3335
type: B, layer: 1, pos: 3267
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 2030
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2045
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 2027
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 2037
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 2036
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 3462
type: B, layer: 1, pos: 2051
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 149
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2069
type: B, layer: 1, pos: 2369
type: B, layer: 1, pos: 2693
type: B, layer: 1, pos: 2694
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 2999
type: B, layer: 1, pos: 3142
type: B, layer: 1, pos: 3145
type: B, layer: 1, pos: 3554

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3562

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0251214, upper bound: 0.0253387
time: 137.21 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0251778, upper bound: 0.0253385
time: 59.10 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.3858655, -0.0662681, -0.3858718, -0.0662646, -0.1044050, 0.1041387
1: -2.3532519, -1.6717043, -2.3532569, -1.6715183, -0.1548315, 0.1543625
2: -1.7063278, -1.2951304, -1.7066450, -1.2951255, -0.1016033, 0.1019759
3: -2.5234873, -2.0174785, -2.5234940, -2.0174041, -0.3196663, 0.3203369
4: -0.5591605, -0.3382534, -0.5592613, -0.3382530, -0.0938092, 0.0938773
5: -2.7547369, -2.1723540, -2.7547500, -2.1723194, -0.3831821, 0.3831339
6: -6.0937061, -5.2512035, -6.0937085, -5.2509303, -0.2199512, 0.2203114
7: -0.5137160, 0.0605588, -0.5139398, 0.0605560, -0.1387139, 0.1396914
8: -0.4112105, -0.0150931, -0.4112044, -0.0149805, -0.1573139, 0.1576568
9: 0.0314196, 0.3173108, 0.0312453, 0.3173135, -0.0896939, 0.0893421

Time for backsubstitution: 5.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3562
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 3553
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 408
type: B, layer: 1, pos: 3566
type: B, layer: 1, pos: 391
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2434
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 3081
type: B, layer: 1, pos: 3300
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 3051
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2086
type: B, layer: 1, pos: 3017
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 2411
type: B, layer: 1, pos: 3576
type: B, layer: 1, pos: 2070
type: B, layer: 1, pos: 3540
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 3041
type: B, layer: 1, pos: 2391
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 3034
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 2972
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2297
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 2845
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2395
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 3338
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2028
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 3008
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2029
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2046
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 3122
type: B, layer: 1, pos: 3335
type: B, layer: 1, pos: 3267
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 2030
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2045
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 2027
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 2037
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 2036
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 3462
type: B, layer: 1, pos: 2051
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 149
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2069
type: B, layer: 1, pos: 2369
type: B, layer: 1, pos: 2693
type: B, layer: 1, pos: 2694
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 2999
type: B, layer: 1, pos: 3142
type: B, layer: 1, pos: 3145
type: B, layer: 1, pos: 3554

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3562

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0250971, upper bound: 0.0253385
time: 98.05 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0251567, upper bound: 0.0253399
time: 106.53 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.3861325, -0.0662852, -0.3858873, -0.0662784, -0.1045012, 0.1045635
1: -2.3558969, -1.6710992, -2.3532629, -1.6711409, -0.1578728, 0.1546373
2: -1.7074087, -1.2904547, -1.7073517, -1.2951210, -0.1020599, 0.1073361
3: -2.5245383, -2.0173807, -2.5235076, -2.0173743, -0.3209394, 0.3204554
4: -0.5595452, -0.3368002, -0.5594730, -0.3382527, -0.0940312, 0.0955611
5: -2.7555513, -2.1723192, -2.7547750, -2.1723232, -0.3841246, 0.3834023
6: -6.0974979, -5.2507067, -6.0937076, -5.2507067, -0.2247604, 0.2205542
7: -0.5144412, 0.0635353, -0.5144149, 0.0605576, -0.1389749, 0.1430935
8: -0.4130376, -0.0144886, -0.4112048, -0.0147351, -0.1593886, 0.1578933
9: 0.0308659, 0.3197039, 0.0308763, 0.3173176, -0.0900315, 0.0917898

Time for backsubstitution: 5.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3562
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 3553
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 408
type: B, layer: 1, pos: 3566
type: B, layer: 1, pos: 391
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2434
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 3081
type: B, layer: 1, pos: 3300
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 3051
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2086
type: B, layer: 1, pos: 3017
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 2411
type: B, layer: 1, pos: 3576
type: B, layer: 1, pos: 2070
type: B, layer: 1, pos: 3540
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 3041
type: B, layer: 1, pos: 2391
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 3034
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 2972
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2297
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 2845
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2395
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 3338
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2028
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 3008
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2029
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2046
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 3122
type: B, layer: 1, pos: 3335
type: B, layer: 1, pos: 3267
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 2030
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 2045
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 2027
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 2037
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 2036
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 3462
type: B, layer: 1, pos: 2051
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 149
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2054
type: B, layer: 1, pos: 2069
type: B, layer: 1, pos: 2369
type: B, layer: 1, pos: 2693
type: B, layer: 1, pos: 2694
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 2999
type: B, layer: 1, pos: 3142
type: B, layer: 1, pos: 3145
type: B, layer: 1, pos: 3554

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3562

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0252818, upper bound: 0.0251646
time: 41.85 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0253371, upper bound: 0.0251641
time: 41.93 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 89.35 seconds
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 89.35
Output dim: 9, lower bound: -0.0252818, upper bound: 0.0251657
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 89.35
Output dim: 9, lower bound: -0.0253390, upper bound: 0.0251630
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 89.35
Output dim: 9, lower bound: -0.0249501, upper bound: 0.0253416
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 89.35
Output dim: 9, lower bound: -0.0250090, upper bound: 0.0253381
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 89.35
Output dim: 9, lower bound: -0.0251214, upper bound: 0.0253387
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 89.35
Output dim: 9, lower bound: -0.0251778, upper bound: 0.0253385
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 89.35
Output dim: 9, lower bound: -0.0250971, upper bound: 0.0253385
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 89.35
Output dim: 9, lower bound: -0.0251567, upper bound: 0.0253399
NS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 89.35
Output dim: 9, lower bound: -0.0252818, upper bound: 0.0251646
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 89.35
Output dim: 9, lower bound: -0.0253371, upper bound: 0.0251641

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 61.94 + 1765.29 = 1827.23 seconds
