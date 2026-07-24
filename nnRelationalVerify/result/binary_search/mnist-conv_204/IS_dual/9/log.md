## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 1.61680046168
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.5487733, 4.5487733)
1: (-17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203)
2: (-8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.9031115, 3.9031115)
3: (-13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582)
4: (-3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.6634710, 3.6634710)
5: (-13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.8109388, 3.8109393)
6: (-15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.5555229, 4.5555229)
7: (-8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024)
8: (-6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597)
9: (3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937)

## BASE Result
execution time: IAR + LP analysis = 15.61 + 34.50 = 50.11 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -2.0738973, upper bound: 2.0738940


# Binary Search by BASE starts (time budget: 3549.89 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.6592936515808105
rel_dist={9: [-1.6176245637311917, 1.6176249981850015]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.VERIFIED, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.6068572998046875
rel_dist={9: [-1.2926447051799563, 1.2926467080621613]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=4, k_high=5, k_mid=4, eps_mid=0.0156250, abs_max=2.6592936515808105
rel_dist={9: [-1.417875488335799, 1.4178760636054193]}

## Binary search (step 3) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=5, k_high=5, k_mid=5, eps_mid=0.0195312, abs_max=2.6592936515808105
rel_dist={9: [-1.5239616905627367, 1.523961546591452]}

## Binary Search Result
Binary search time: 209.42 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01953125


# Individual Split (IS_dual) starts
Time budget: 3340.46 seconds

## Binary search (step 0) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5875
type: A, layer: 1, pos: 5875
type: A, layer: 1, pos: 4557
type: B, layer: 1, pos: 4557
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 4597
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 5820
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5875

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8315608, upper bound: 1.8581702
time: 5.01 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8604469, upper bound: 1.8604473
time: 4.99 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 10.22 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 10.22
Output dim: 9, lower bound: -1.8315608, upper bound: 1.8581702
IS_B2, status: Status.UNKNOWN, split count: 1, time: 10.22
Output dim: 9, lower bound: -1.8604469, upper bound: 1.8604473

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -9.7076283, -5.1739035, -9.6517572, -5.2123337, -3.9833193, 3.9657006
1: -17.1163940, -13.3358536, -17.0863571, -13.3565741, -3.7598200, 3.7505035
2: -8.1562595, -4.2671590, -8.1017151, -4.3077283, -3.4805641, 3.4665751
3: -13.8203630, -8.7370911, -13.7874165, -8.7640533, -5.0563097, 5.0503254
4: -3.9069490, -0.2473956, -3.8931594, -0.2618493, -3.6069317, 3.6062307
5: -13.9862242, -9.9761734, -13.9540033, -9.9989052, -3.3240514, 3.3139591
6: -15.9506664, -11.4054737, -15.9145575, -11.4365091, -4.1076736, 4.1027708
7: -8.3875217, -4.1859102, -8.3611498, -4.2133913, -4.1741304, 4.1752396
8: -6.7115211, -2.9894505, -6.6736360, -3.0213003, -3.6902208, 3.6841855
9: 3.9072409, 6.5584183, 3.9322205, 6.5280085, -2.6207676, 2.6261978

Time for backsubstitution: 14.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4557
type: B, layer: 1, pos: 4557
type: A, layer: 1, pos: 5875
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 4597
type: B, layer: 1, pos: 5820
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4557

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8314930, upper bound: 1.8318922
time: 4.58 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8314935, upper bound: 1.8581029
time: 4.86 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -9.7212811, -5.1725097, -9.7212772, -5.1725125, -4.1678419, 4.1626401
1: -17.1234474, -13.3353291, -17.1234398, -13.3353319, -3.7881155, 3.7881107
2: -8.1695728, -4.2664623, -8.1695595, -4.2664623, -3.6593046, 3.6541162
3: -13.8282661, -8.7362089, -13.8282642, -8.7362051, -5.0920610, 5.0920553
4: -3.9074690, -0.2440012, -3.9074695, -0.2440007, -3.6288662, 3.6350193
5: -13.9941177, -9.9756889, -13.9941082, -9.9756899, -3.3869648, 3.3817687
6: -15.9595585, -11.4040346, -15.9595528, -11.4040365, -4.2286806, 4.2238479
7: -8.3881950, -4.1792917, -8.3881950, -4.1793032, -4.2088919, 4.2089033
8: -6.7207699, -2.9882140, -6.7207656, -2.9882154, -3.7325544, 3.7325516
9: 3.9066472, 6.5659409, 3.9066486, 6.5659370, -2.6592898, 2.6592922

Time for backsubstitution: 14.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4557
type: A, layer: 1, pos: 4557
type: A, layer: 1, pos: 5875
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 4597
type: B, layer: 1, pos: 5820
type: A, layer: 1, pos: 5820
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4557

## Relational analysis of IS_B2_B1

### Relational analysis result of IS_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8341422, upper bound: 1.8603806
time: 4.42 seconds

## Relational analysis of IS_B2_B2

### Relational analysis result of IS_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8603787, upper bound: 1.8603802
time: 4.78 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 23.99 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 23.99
Output dim: 9, lower bound: -1.8314930, upper bound: 1.8318922
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 23.99
Output dim: 9, lower bound: -1.8314935, upper bound: 1.8581029
IS_B2_B1, status: Status.UNKNOWN, split count: 2, time: 23.99
Output dim: 9, lower bound: -1.8341422, upper bound: 1.8603806
IS_B2_B2, status: Status.UNKNOWN, split count: 2, time: 23.99
Output dim: 9, lower bound: -1.8603787, upper bound: 1.8603802

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -9.6883631, -5.1809926, -9.6481819, -5.2132645, -3.9652100, 3.9440403
1: -17.0900078, -13.3744850, -17.0814133, -13.3621302, -3.7278776, 3.7069283
2: -8.1042118, -4.2959070, -8.0908909, -4.3107691, -3.4279423, 3.4262137
3: -13.7482491, -8.7574615, -13.7726631, -8.7662458, -4.9820032, 5.0152016
4: -3.8353460, -0.2949450, -3.8792372, -0.2679958, -3.5373230, 3.5842922
5: -13.9581175, -10.0154495, -13.9507809, -10.0062990, -3.2920365, 3.2766781
6: -15.9268389, -11.4409285, -15.9115047, -11.4439316, -4.0555353, 4.0465355
7: -8.3219719, -4.2857313, -8.3503389, -4.2319884, -4.0899835, 4.0646076
8: -6.6689262, -3.0324993, -6.6683893, -3.0296516, -3.6392746, 3.6358900
9: 3.9831934, 6.5046940, 3.9409103, 6.5174379, -2.5342445, 2.5637836

Time for backsubstitution: 14.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5875
type: B, layer: 1, pos: 4557
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4597
type: B, layer: 1, pos: 5820
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 494

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5875

## Relational analysis of IS_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8314930, upper bound: 1.8053134
time: 4.64 seconds

## Relational analysis of IS_B1_A1_A2

### Relational analysis result of IS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8314930, upper bound: 1.8318922
time: 4.47 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -9.7076111, -5.1739049, -9.6517572, -5.2123337, -3.9839783, 3.9644508
1: -17.1163864, -13.3358660, -17.0863571, -13.3565741, -3.7598124, 3.7504911
2: -8.1562300, -4.2671652, -8.1017151, -4.3077283, -3.4586620, 3.4665685
3: -13.8203268, -8.7370939, -13.7874165, -8.7640533, -5.0562735, 5.0503225
4: -3.9069185, -0.2474124, -3.8931594, -0.2618493, -3.5865993, 3.6062136
5: -13.9862080, -9.9761801, -13.9540033, -9.9989052, -3.3240457, 3.3070240
6: -15.9506598, -11.4054956, -15.9145575, -11.4365091, -4.1002188, 4.1157103
7: -8.3874989, -4.1859493, -8.3611498, -4.2133913, -4.1741076, 4.1752005
8: -6.7115073, -2.9894705, -6.6736360, -3.0213003, -3.6902070, 3.6841655
9: 3.9072618, 6.5583878, 3.9322205, 6.5280085, -2.6207466, 2.6261673

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5875
type: B, layer: 1, pos: 4557
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 5820
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 5820
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 494

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5875

## Relational analysis of IS_B1_A2_A1

### Relational analysis result of IS_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8314935, upper bound: 1.8314948
time: 4.71 seconds

## Relational analysis of IS_B1_A2_A2

### Relational analysis result of IS_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8314935, upper bound: 1.8581029
time: 4.80 seconds

## BFS IS instance: IS_B2_B1

### Backsubstitution after applying IS history:
0: -9.7177124, -5.1734428, -9.7020168, -5.1795983, -4.1459112, 4.1445532
1: -17.1185055, -13.3408966, -17.0970554, -13.3739710, -3.7445345, 3.7561588
2: -8.1587486, -4.2695093, -8.1175184, -4.2952156, -3.6188722, 3.6014228
3: -13.8135223, -8.7384005, -13.7561626, -8.7565775, -5.0569448, 5.0177622
4: -3.8935475, -0.2501336, -3.8358641, -0.2915483, -3.6019993, 3.5654154
5: -13.9908915, -9.9830828, -13.9660082, -10.0149679, -3.3495684, 3.3496909
6: -15.9565010, -11.4114761, -15.9357290, -11.4394979, -4.1723671, 4.1716423
7: -8.3773518, -4.1978865, -8.3226089, -4.2791243, -4.0982275, 4.1247225
8: -6.7155237, -2.9965773, -6.6781731, -3.0312710, -3.6842527, 3.6815958
9: 3.9153605, 6.5553689, 3.9826174, 6.5122128, -2.5968523, 2.5727515

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5875
type: A, layer: 1, pos: 4557
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 4597
type: B, layer: 1, pos: 5820
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 494

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5875

## Relational analysis of IS_B2_B1_A1

### Relational analysis result of IS_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053082, upper bound: 1.8314925
time: 4.67 seconds

## Relational analysis of IS_B2_B1_A2

### Relational analysis result of IS_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053083, upper bound: 1.8314936
time: 6.16 seconds

## BFS IS instance: IS_B2_B2

### Backsubstitution after applying IS history:
0: -9.7212811, -5.1725097, -9.7212629, -5.1725130, -4.1665916, 4.1633019
1: -17.1234474, -13.3353291, -17.1234303, -13.3353443, -3.7881031, 3.7881012
2: -8.1695728, -4.2664623, -8.1695328, -4.2664700, -3.6592970, 3.6322136
3: -13.8282661, -8.7362089, -13.8282261, -8.7362127, -5.0920534, 5.0920172
4: -3.9074690, -0.2440012, -3.9074385, -0.2440178, -3.6288476, 3.6146884
5: -13.9941177, -9.9756889, -13.9940996, -9.9756966, -3.3800287, 3.3817616
6: -15.9595585, -11.4040346, -15.9595490, -11.4040546, -4.2416162, 4.2163897
7: -8.3881950, -4.1792917, -8.3881712, -4.1793394, -4.2088556, 4.2088795
8: -6.7207699, -2.9882140, -6.7207527, -2.9882340, -3.7325358, 3.7325387
9: 3.9066472, 6.5659409, 3.9066682, 6.5659094, -2.6592622, 2.6592727

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5875
type: A, layer: 1, pos: 4557
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 5820
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 5820
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 494

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5875

## Relational analysis of IS_B2_B2_A1

### Relational analysis result of IS_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8314928, upper bound: 1.8314924
time: 4.74 seconds

## Relational analysis of IS_B2_B2_A2

### Relational analysis result of IS_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8314923, upper bound: 1.8314925
time: 5.03 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 24.52 seconds
IS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 24.52
Output dim: 9, lower bound: -1.8314930, upper bound: 1.8053134
IS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 24.52
Output dim: 9, lower bound: -1.8314930, upper bound: 1.8318922
IS_B1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 24.52
Output dim: 9, lower bound: -1.8314935, upper bound: 1.8314948
IS_B1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 24.52
Output dim: 9, lower bound: -1.8314935, upper bound: 1.8581029
IS_B2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 24.52
Output dim: 9, lower bound: -1.8053082, upper bound: 1.8314925
IS_B2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 24.52
Output dim: 9, lower bound: -1.8053083, upper bound: 1.8314936
IS_B2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 24.52
Output dim: 9, lower bound: -1.8314928, upper bound: 1.8314924
IS_B2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 24.52
Output dim: 9, lower bound: -1.8314923, upper bound: 1.8314925

## BFS IS instance: IS_B1_A1_A1

### Backsubstitution after applying IS history:
0: -9.6324787, -5.2194176, -9.6481819, -5.2132645, -3.9090843, 3.9056273
1: -17.0599537, -13.3951693, -17.0814133, -13.3621302, -3.6978235, 3.6862440
2: -8.0496616, -4.3364553, -8.0908909, -4.3107691, -3.3733921, 3.3856792
3: -13.7152596, -8.7844372, -13.7726631, -8.7662458, -4.9490137, 4.9882259
4: -3.8215551, -0.3094628, -3.8792372, -0.2679958, -3.5228071, 3.5697744
5: -13.9259109, -10.0381699, -13.9507809, -10.0062990, -3.2594943, 3.2542305
6: -15.8907461, -11.4718838, -15.9115047, -11.4439316, -4.0195312, 4.0154824
7: -8.2959061, -4.3132200, -8.3503389, -4.2319884, -4.0639176, 4.0371189
8: -6.6310477, -3.0642805, -6.6683893, -3.0296516, -3.6013961, 3.6041088
9: 4.0080853, 6.4742885, 3.9409103, 6.5174379, -2.5093527, 2.5333781

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4557
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4597
type: B, layer: 1, pos: 5820
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 494

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4557

## Relational analysis of IS_B1_A1_A1_B1

### Relational analysis result of IS_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053124, upper bound: 1.8053111
time: 4.68 seconds

## Relational analysis of IS_B1_A1_A1_B2

### Relational analysis result of IS_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053124, upper bound: 1.8053113
time: 5.15 seconds

## BFS IS instance: IS_B1_A1_A2

### Backsubstitution after applying IS history:
0: -9.7019901, -5.1808753, -9.6481819, -5.2132645, -3.9789696, 3.9441657
1: -17.0970135, -13.3743057, -17.0814133, -13.3621302, -3.7348833, 3.7071075
2: -8.1175013, -4.2958589, -8.0908909, -4.3107691, -3.4412413, 3.4128430
3: -13.7561131, -8.7570534, -13.7726631, -8.7662458, -4.9898672, 5.0156097
4: -3.8356650, -0.2915885, -3.8792372, -0.2679958, -3.5369072, 3.5876486
5: -13.9660034, -10.0154209, -13.9507809, -10.0062990, -3.3000302, 3.2761173
6: -15.9357185, -11.4404163, -15.9115047, -11.4439316, -4.0644274, 4.0471191
7: -8.3221769, -4.2791514, -8.3503389, -4.2319884, -4.0901885, 4.0711875
8: -6.6781597, -3.0321503, -6.6683893, -3.0296516, -3.6485081, 3.6362391
9: 3.9829741, 6.5122108, 3.9409103, 6.5174379, -2.5344639, 2.5713005

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4557
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4597
type: B, layer: 1, pos: 5820
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 494

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4557

## Relational analysis of IS_B1_A1_A2_B1

### Relational analysis result of IS_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053123, upper bound: 1.8318899
time: 4.54 seconds

## Relational analysis of IS_B1_A1_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053124, upper bound: 1.8318899
time: 4.77 seconds

## BFS IS instance: IS_B1_A2_A1

### Backsubstitution after applying IS history:
0: -9.6517410, -5.2123356, -9.6517572, -5.2123337, -3.9279118, 3.9260302
1: -17.0863514, -13.3565893, -17.0863571, -13.3565741, -3.7297773, 3.7297678
2: -8.1016903, -4.3077345, -8.1017151, -4.3077283, -3.4041157, 3.4260154
3: -13.7873850, -8.7640591, -13.7874165, -8.7640533, -5.0233316, 5.0233574
4: -3.8931270, -0.2618673, -3.8931594, -0.2618493, -3.5720768, 3.5923886
5: -13.9539986, -9.9989147, -13.9540033, -9.9989052, -3.2914834, 3.2845550
6: -15.9145527, -11.4365282, -15.9145575, -11.4365091, -4.0641956, 4.0846109
7: -8.3611259, -4.2134304, -8.3611498, -4.2133913, -4.1477346, 4.1477194
8: -6.6736231, -3.0213170, -6.6736360, -3.0213003, -3.6523228, 3.6523190
9: 3.9322391, 6.5279784, 3.9322205, 6.5280085, -2.5957694, 2.5957580

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4557
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 5820
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 5820
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 494

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4557

## Relational analysis of IS_B1_A2_A1_B1

### Relational analysis result of IS_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053083, upper bound: 1.8314943
time: 4.71 seconds

## Relational analysis of IS_B1_A2_A1_B2

### Relational analysis result of IS_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053082, upper bound: 1.8314960
time: 4.88 seconds

## BFS IS instance: IS_B1_A2_A2

### Backsubstitution after applying IS history:
0: -9.7212362, -5.1737866, -9.6517572, -5.2123337, -3.9977169, 3.9645743
1: -17.1233864, -13.3356800, -17.0863571, -13.3565741, -3.7668123, 3.7506771
2: -8.1695175, -4.2671132, -8.1017151, -4.3077283, -3.4719658, 3.4542408
3: -13.8281803, -8.7366867, -13.7874165, -8.7640533, -5.0641270, 5.0507298
4: -3.9072380, -0.2440574, -3.8931594, -0.2618493, -3.5861893, 3.6101532
5: -13.9940891, -9.9761477, -13.9540033, -9.9989052, -3.3320413, 3.3064847
6: -15.9595337, -11.4049778, -15.9145575, -11.4365091, -4.1091080, 4.1163092
7: -8.3877459, -4.1793680, -8.3611498, -4.2133913, -4.1743546, 4.1817818
8: -6.7207437, -2.9891171, -6.6736360, -3.0213003, -3.6994433, 3.6845188
9: 3.9070263, 6.5659070, 3.9322205, 6.5280085, -2.6209822, 2.6336865

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4557
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 5820
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 5820
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 494

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4557

## Relational analysis of IS_B1_A2_A2_B1

### Relational analysis result of IS_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053083, upper bound: 1.8581018
time: 4.60 seconds

## Relational analysis of IS_B1_A2_A2_B2

### Relational analysis result of IS_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053083, upper bound: 1.8581035
time: 5.26 seconds

## BFS IS instance: IS_B2_B1_A1

### Backsubstitution after applying IS history:
0: -9.6481819, -5.2132645, -9.7020168, -5.1795983, -4.0757284, 3.9790850
1: -17.0814133, -13.3621302, -17.0970554, -13.3739710, -3.7074423, 3.7349253
2: -8.0908909, -4.3107691, -8.1175184, -4.2952156, -3.5440435, 3.4412599
3: -13.7726631, -8.7662458, -13.7561626, -8.7565775, -5.0160856, 4.9899168
4: -3.8792372, -0.2679958, -3.8358641, -0.2915483, -3.5876889, 3.5468035
5: -13.9507809, -10.0062990, -13.9660082, -10.0149679, -3.3088598, 3.3000369
6: -15.9115047, -11.4439316, -15.9357290, -11.4394979, -4.1274014, 4.0644627
7: -8.3503389, -4.2319884, -8.3226089, -4.2791243, -4.0712147, 4.0906205
8: -6.6683893, -3.0296516, -6.6781731, -3.0312710, -3.6371183, 3.6485214
9: 3.9409103, 6.5174379, 3.9826174, 6.5122128, -2.5713024, 2.5348206

Time for backsubstitution: 14.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4557
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 4597
type: B, layer: 1, pos: 5820
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 5820
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 494

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4557

## Relational analysis of IS_B2_B1_A1_A1

### Relational analysis result of IS_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053082, upper bound: 1.8053115
time: 4.71 seconds

## Relational analysis of IS_B2_B1_A1_A2

### Relational analysis result of IS_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053082, upper bound: 1.8314925
time: 4.73 seconds

## BFS IS instance: IS_B2_B1_A2

### Backsubstitution after applying IS history:
0: -9.7177067, -5.1734443, -9.7020168, -5.1795983, -4.1410856, 4.1445527
1: -17.1184940, -13.3408947, -17.0970554, -13.3739710, -3.7445230, 3.7561607
2: -8.1587372, -4.2695093, -8.1175184, -4.2952156, -3.6188722, 3.6066022
3: -13.8135185, -8.7384005, -13.7561626, -8.7565775, -5.0569410, 5.0177622
4: -3.8935461, -0.2501363, -3.8358641, -0.2915483, -3.6019979, 3.5592608
5: -13.9908819, -9.9830856, -13.9660082, -10.0149679, -3.3443727, 3.3496904
6: -15.9565001, -11.4114771, -15.9357290, -11.4394979, -4.1675310, 4.1716418
7: -8.3773527, -4.1978941, -8.3226089, -4.2791243, -4.0982285, 4.1247149
8: -6.7155199, -2.9965792, -6.6781731, -3.0312710, -3.6842489, 3.6815939
9: 3.9153605, 6.5553665, 3.9826174, 6.5122128, -2.5968523, 2.5727491

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4557
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 4597
type: B, layer: 1, pos: 5820
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 494

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4557

## Relational analysis of IS_B2_B1_A2_A1

### Relational analysis result of IS_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053083, upper bound: 1.8341480
time: 4.72 seconds

## Relational analysis of IS_B2_B1_A2_A2

### Relational analysis result of IS_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053083, upper bound: 1.8314936
time: 6.14 seconds

## BFS IS instance: IS_B2_B2_A1

### Backsubstitution after applying IS history:
0: -9.6517572, -5.2123337, -9.7212629, -5.1725130, -4.0964718, 3.9978333
1: -17.0863571, -13.3565741, -17.1234303, -13.3353443, -3.7510128, 3.7668562
2: -8.1017151, -4.3077283, -8.1695328, -4.2664700, -3.5842600, 3.4719839
3: -13.7874165, -8.7640533, -13.8282261, -8.7362127, -5.0512037, 5.0641727
4: -3.8931594, -0.2618493, -3.9074385, -0.2440178, -3.6106944, 3.5960789
5: -13.9540033, -9.9989052, -13.9940996, -9.9756966, -3.3393192, 3.3320475
6: -15.9145575, -11.4365091, -15.9595490, -11.4040546, -4.1966648, 4.1091447
7: -8.3611498, -4.2133913, -8.3881712, -4.1793394, -4.1818104, 4.1747799
8: -6.6736360, -3.0213003, -6.7207527, -2.9882340, -3.6854019, 3.6994524
9: 3.9322205, 6.5280085, 3.9066682, 6.5659094, -2.6336889, 2.6213403

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4557
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 5820
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 5820
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4557

## Relational analysis of IS_B2_B2_A1_A1

### Relational analysis result of IS_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053082, upper bound: 1.8053082
time: 4.97 seconds

## Relational analysis of IS_B2_B2_A1_A2

### Relational analysis result of IS_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053083, upper bound: 1.8314928
time: 4.90 seconds

## BFS IS instance: IS_B2_B2_A2

### Backsubstitution after applying IS history:
0: -9.7212772, -5.1725125, -9.7212629, -5.1725130, -4.1613894, 4.1633029
1: -17.1234398, -13.3353319, -17.1234303, -13.3353443, -3.7880955, 3.7880983
2: -8.1695595, -4.2664623, -8.1695328, -4.2664700, -3.6592894, 3.6373925
3: -13.8282642, -8.7362051, -13.8282261, -8.7362127, -5.0920515, 5.0920210
4: -3.9074695, -0.2440007, -3.9074385, -0.2440178, -3.6288476, 3.6085343
5: -13.9941082, -9.9756899, -13.9940996, -9.9756966, -3.3748331, 3.3817616
6: -15.9595528, -11.4040365, -15.9595490, -11.4040546, -4.2367802, 4.2163887
7: -8.3881950, -4.1793032, -8.3881712, -4.1793394, -4.2088556, 4.2088680
8: -6.7207656, -2.9882154, -6.7207527, -2.9882340, -3.7325315, 3.7325373
9: 3.9066486, 6.5659370, 3.9066682, 6.5659094, -2.6592607, 2.6592689

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4557
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 5820
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 5820
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 494

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4557

## Relational analysis of IS_B2_B2_A2_A1

### Relational analysis result of IS_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053083, upper bound: 1.8341430
time: 5.15 seconds

## Relational analysis of IS_B2_B2_A2_A2

### Relational analysis result of IS_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053083, upper bound: 1.8314936
time: 4.83 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 24.71 seconds
IS_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.71
Output dim: 9, lower bound: -1.8053124, upper bound: 1.8053111
IS_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.71
Output dim: 9, lower bound: -1.8053124, upper bound: 1.8053113
IS_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.71
Output dim: 9, lower bound: -1.8053123, upper bound: 1.8318899
IS_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.71
Output dim: 9, lower bound: -1.8053124, upper bound: 1.8318899
IS_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.71
Output dim: 9, lower bound: -1.8053083, upper bound: 1.8314943
IS_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.71
Output dim: 9, lower bound: -1.8053082, upper bound: 1.8314960
IS_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.71
Output dim: 9, lower bound: -1.8053083, upper bound: 1.8581018
IS_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.71
Output dim: 9, lower bound: -1.8053083, upper bound: 1.8581035
IS_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 24.71
Output dim: 9, lower bound: -1.8053082, upper bound: 1.8053115
IS_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 24.71
Output dim: 9, lower bound: -1.8053082, upper bound: 1.8314925
IS_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 24.71
Output dim: 9, lower bound: -1.8053083, upper bound: 1.8341480
IS_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 24.71
Output dim: 9, lower bound: -1.8053083, upper bound: 1.8314936
IS_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 24.71
Output dim: 9, lower bound: -1.8053082, upper bound: 1.8053082
IS_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 24.71
Output dim: 9, lower bound: -1.8053083, upper bound: 1.8314928
IS_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 24.71
Output dim: 9, lower bound: -1.8053083, upper bound: 1.8341430
IS_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 24.71
Output dim: 9, lower bound: -1.8053083, upper bound: 1.8314936

## BFS IS instance: IS_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -9.6324787, -5.2194176, -9.6324787, -5.2194176, -3.8891821, 3.8891821
1: -17.0599537, -13.3951693, -17.0599537, -13.3951693, -3.6647844, 3.6647844
2: -8.0496616, -4.3364553, -8.0496616, -4.3364553, -3.3469429, 3.3469429
3: -13.7152596, -8.7844372, -13.7152596, -8.7844372, -4.9308224, 4.9308224
4: -3.8215551, -0.3094628, -3.8215551, -0.3094628, -3.5120924, 3.5120924
5: -13.9259109, -10.0381699, -13.9259109, -10.0381699, -3.2304430, 3.2304435
6: -15.8907461, -11.4718838, -15.8907461, -11.4718838, -3.9868813, 3.9868803
7: -8.2959061, -4.3132200, -8.2959061, -4.3132200, -3.9826860, 3.9826860
8: -6.6310477, -3.0642805, -6.6310477, -3.0642805, -3.5667672, 3.5667672
9: 4.0080853, 6.4742885, 4.0080853, 6.4742885, -2.4662032, 2.4662032

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 5820
type: B, layer: 1, pos: 5820
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 961

## Relational analysis of IS_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4597

## Relational analysis of IS_B1_A1_A1_B1_B1

### Relational analysis result of IS_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8051938, upper bound: 1.8040164
time: 4.36 seconds

## Relational analysis of IS_B1_A1_A1_B1_B2

### Relational analysis result of IS_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053152, upper bound: 1.8053108
time: 4.57 seconds

## BFS IS instance: IS_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -9.6324787, -5.2194176, -9.6516857, -5.2123361, -3.9085884, 3.9103112
1: -17.0599537, -13.3951693, -17.0863495, -13.3565950, -3.7033587, 3.6911802
2: -8.0496616, -4.3364553, -8.1016855, -4.3077364, -3.3765850, 3.3960733
3: -13.7152596, -8.7844372, -13.7873831, -8.7640629, -4.9511967, 5.0029459
4: -3.8215551, -0.3094628, -3.8931284, -0.2618936, -3.5288191, 3.5836656
5: -13.9259109, -10.0381699, -13.9539986, -9.9989195, -3.2670193, 3.2567749
6: -15.8907461, -11.4718838, -15.9145517, -11.4365759, -4.0260096, 4.0165377
7: -8.2959061, -4.3132200, -8.3611193, -4.2134285, -4.0824776, 4.0478992
8: -6.6310477, -3.0642805, -6.6736207, -3.0213208, -3.6097269, 3.6093402
9: 4.0080853, 6.4742885, 3.9322443, 6.5279779, -2.5198927, 2.5420442

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4597
type: B, layer: 1, pos: 5820
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 494

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 961

## Relational analysis of IS_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4597

## Relational analysis of IS_B1_A1_A1_B2_B1

### Relational analysis result of IS_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8051938, upper bound: 1.8040165
time: 5.90 seconds

## Relational analysis of IS_B1_A1_A1_B2_B2

### Relational analysis result of IS_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053152, upper bound: 1.8053111
time: 5.23 seconds

## BFS IS instance: IS_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -9.7019901, -5.1808753, -9.6324787, -5.2194176, -3.9590969, 3.9277201
1: -17.0970135, -13.3743057, -17.0599537, -13.3951693, -3.7018442, 3.6856480
2: -8.1175013, -4.2958589, -8.0496616, -4.3364553, -3.4147921, 3.3838577
3: -13.7561131, -8.7570534, -13.7152596, -8.7844372, -4.9716759, 4.9582062
4: -3.8356650, -0.2915885, -3.8215551, -0.3094628, -3.5262022, 3.5299666
5: -13.9660034, -10.0154209, -13.9259109, -10.0381699, -3.2709799, 3.2523308
6: -15.9357185, -11.4404163, -15.8907461, -11.4718838, -4.0317755, 4.0185175
7: -8.3221769, -4.2791514, -8.2959061, -4.3132200, -4.0089569, 4.0167546
8: -6.6781597, -3.0321503, -6.6310477, -3.0642805, -3.6138792, 3.5988975
9: 3.9829741, 6.5122108, 4.0080853, 6.4742885, -2.4913144, 2.5041256

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 4597
type: B, layer: 1, pos: 5820
type: A, layer: 1, pos: 5820
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 494

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 961

## Relational analysis of IS_B1_A1_A2_B1_B1

### Relational analysis result of IS_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7880005, upper bound: 1.8277246
time: 4.55 seconds

## Relational analysis of IS_B1_A1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4597

## Relational analysis of IS_B1_A1_A2_B1_B1

### Relational analysis result of IS_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8051907, upper bound: 1.8305902
time: 4.44 seconds

## Relational analysis of IS_B1_A1_A2_B1_B2

### Relational analysis result of IS_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053121, upper bound: 1.8318886
time: 4.91 seconds

## BFS IS instance: IS_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -9.7019901, -5.1808753, -9.6516857, -5.2123361, -3.9784718, 3.9473667
1: -17.0970135, -13.3743057, -17.0863495, -13.3565950, -3.7404184, 3.7120438
2: -8.1175013, -4.2958589, -8.1016855, -4.3077364, -3.4444342, 3.4166138
3: -13.7561131, -8.7570534, -13.7873831, -8.7640629, -4.9920502, 5.0303297
4: -3.8356650, -0.2915885, -3.8931284, -0.2618936, -3.5429201, 3.6015399
5: -13.9660034, -10.0154209, -13.9539986, -9.9989195, -3.3075552, 3.2786617
6: -15.9357185, -11.4404163, -15.9145517, -11.4365759, -4.0709047, 4.0481749
7: -8.3221769, -4.2791514, -8.3611193, -4.2134285, -4.1087484, 4.0819678
8: -6.6781597, -3.0321503, -6.6736207, -3.0213208, -3.6568389, 3.6414704
9: 3.9829741, 6.5122108, 3.9322443, 6.5279779, -2.5450039, 2.5799665

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4597
type: B, layer: 1, pos: 5820
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 494

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 961

## Relational analysis of IS_B1_A1_A2_B2_B1

### Relational analysis result of IS_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7880006, upper bound: 1.8277239
time: 5.50 seconds

## Relational analysis of IS_B1_A1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4597

## Relational analysis of IS_B1_A1_A2_B2_B1

### Relational analysis result of IS_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8051907, upper bound: 1.8305898
time: 6.66 seconds

## Relational analysis of IS_B1_A1_A2_B2_B2

### Relational analysis result of IS_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053122, upper bound: 1.8318888
time: 4.95 seconds

## BFS IS instance: IS_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -9.6517410, -5.2123356, -9.6324787, -5.2194176, -3.9105692, 3.9085884
1: -17.0863514, -13.3565893, -17.0599537, -13.3951693, -3.6911821, 3.7033644
2: -8.1016903, -4.3077345, -8.0496616, -4.3364553, -3.3960762, 3.3764286
3: -13.7873850, -8.7640591, -13.7152596, -8.7844372, -5.0029478, 4.9512005
4: -3.8931270, -0.2618673, -3.8215551, -0.3094628, -3.5836642, 3.5255713
5: -13.9539986, -9.9989147, -13.9259109, -10.0381699, -3.2567759, 3.2651429
6: -15.9145527, -11.4365282, -15.8907461, -11.4718838, -4.0165386, 4.0345740
7: -8.3611259, -4.2134304, -8.2959061, -4.3132200, -4.0479059, 4.0824757
8: -6.6736231, -3.0213170, -6.6310477, -3.0642805, -3.6093426, 3.6097307
9: 3.9322391, 6.5279784, 4.0080853, 6.4742885, -2.5420494, 2.5198932

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 4597
type: B, layer: 1, pos: 5820
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 494

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 961

## Relational analysis of IS_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4597

## Relational analysis of IS_B1_A2_A1_B1_A1

### Relational analysis result of IS_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8040162, upper bound: 1.8314877
time: 4.43 seconds

## Relational analysis of IS_B1_A2_A1_B1_A2

### Relational analysis result of IS_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053110, upper bound: 1.8314907
time: 4.86 seconds

## BFS IS instance: IS_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -9.6517410, -5.2123356, -9.6517410, -5.2123356, -3.9279051, 3.9279056
1: -17.0863514, -13.3565893, -17.0863514, -13.3565893, -3.7297621, 3.7297621
2: -8.1016903, -4.3077345, -8.1016903, -4.3077345, -3.4041090, 3.4041090
3: -13.7873850, -8.7640591, -13.7873850, -8.7640591, -5.0233259, 5.0233259
4: -3.8931270, -0.2618673, -3.8931270, -0.2618673, -3.5720577, 3.5720572
5: -13.9539986, -9.9989147, -13.9539986, -9.9989147, -3.2845483, 3.2845478
6: -15.9145527, -11.4365282, -15.9145527, -11.4365282, -4.0845890, 4.0845890
7: -8.3611259, -4.2134304, -8.3611259, -4.2134304, -4.1476955, 4.1476955
8: -6.6736231, -3.0213170, -6.6736231, -3.0213170, -3.6523061, 3.6523061
9: 3.9322391, 6.5279784, 3.9322391, 6.5279784, -2.5957394, 2.5957394

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 4597
type: B, layer: 1, pos: 5820
type: A, layer: 1, pos: 5820
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 961

## Relational analysis of IS_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4597

## Relational analysis of IS_B1_A2_A1_B2_B1

### Relational analysis result of IS_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8051896, upper bound: 1.8303202
time: 5.46 seconds

## Relational analysis of IS_B1_A2_A1_B2_B2

### Relational analysis result of IS_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053110, upper bound: 1.8314924
time: 5.40 seconds

## BFS IS instance: IS_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -9.7212362, -5.1737866, -9.6324787, -5.2194176, -3.9804277, 3.9471288
1: -17.1233864, -13.3356800, -17.0599537, -13.3951693, -3.7282171, 3.7242737
2: -8.1695175, -4.2671132, -8.0496616, -4.3364553, -3.4585323, 3.4048190
3: -13.8281803, -8.7366867, -13.7152596, -8.7844372, -5.0437431, 4.9785728
4: -3.9072380, -0.2440574, -3.8215551, -0.3094628, -3.5977752, 3.5433354
5: -13.9940891, -9.9761477, -13.9259109, -10.0381699, -3.2973337, 3.2870722
6: -15.9595337, -11.4049778, -15.8907461, -11.4718838, -4.0614548, 4.0662818
7: -8.3877459, -4.1793680, -8.2959061, -4.3132200, -4.0745258, 4.1165380
8: -6.7207437, -2.9891171, -6.6310477, -3.0642805, -3.6564631, 3.6419306
9: 3.9070263, 6.5659070, 4.0080853, 6.4742885, -2.5672622, 2.5578218

Time for backsubstitution: 17.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 4597
type: B, layer: 1, pos: 5820
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 494

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 961

## Relational analysis of IS_B1_A2_A2_B1_B1

### Relational analysis result of IS_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7879965, upper bound: 1.8539459
time: 4.66 seconds

## Relational analysis of IS_B1_A2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4597

## Relational analysis of IS_B1_A2_A2_B1_A1

### Relational analysis result of IS_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8040132, upper bound: 1.8580939
time: 4.61 seconds

## Relational analysis of IS_B1_A2_A2_B1_A2

### Relational analysis result of IS_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053080, upper bound: 1.8580974
time: 5.04 seconds

## BFS IS instance: IS_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -9.7212362, -5.1737866, -9.6517410, -5.2123356, -3.9977102, 3.9664464
1: -17.1233864, -13.3356800, -17.0863514, -13.3565893, -3.7667971, 3.7506714
2: -8.1695175, -4.2671132, -8.1016903, -4.3077345, -3.4719601, 3.4409137
3: -13.8281803, -8.7366867, -13.7873850, -8.7640591, -5.0641212, 5.0506983
4: -3.9072380, -0.2440574, -3.8931270, -0.2618673, -3.5861702, 3.5898218
5: -13.9940891, -9.9761477, -13.9539986, -9.9989147, -3.3251061, 3.3064780
6: -15.9595337, -11.4049778, -15.9145527, -11.4365282, -4.1294842, 4.1162872
7: -8.3877459, -4.1793680, -8.3611259, -4.2134304, -4.1743155, 4.1817579
8: -6.7207437, -2.9891171, -6.6736231, -3.0213170, -3.6994267, 3.6845059
9: 3.9070263, 6.5659070, 3.9322391, 6.5279784, -2.6209521, 2.6336679

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 4597
type: B, layer: 1, pos: 5820
type: A, layer: 1, pos: 5820
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 494

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 961

## Relational analysis of IS_B1_A2_A2_B2_B1

### Relational analysis result of IS_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7879965, upper bound: 1.8539468
time: 5.36 seconds

## Relational analysis of IS_B1_A2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4597

## Relational analysis of IS_B1_A2_A2_B2_B1

### Relational analysis result of IS_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8051867, upper bound: 1.8569266
time: 5.05 seconds

## Relational analysis of IS_B1_A2_A2_B2_B2

### Relational analysis result of IS_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053081, upper bound: 1.8580986
time: 4.96 seconds

## BFS IS instance: IS_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -9.6324787, -5.2194176, -9.7020168, -5.1795983, -4.0596552, 3.9592133
1: -17.0599537, -13.3951693, -17.0970554, -13.3739710, -3.6859827, 3.7018862
2: -8.0496616, -4.3364553, -8.1175184, -4.2952156, -3.5052404, 3.4148107
3: -13.7152596, -8.7844372, -13.7561626, -8.7565775, -4.9586821, 4.9717255
4: -3.8215551, -0.3094628, -3.8358641, -0.2915483, -3.5300069, 3.5264013
5: -13.9259109, -10.0381699, -13.9660082, -10.0149679, -3.2850447, 3.2709875
6: -15.8907461, -11.4718838, -15.9357290, -11.4394979, -4.0987711, 4.0318112
7: -8.2959061, -4.3132200, -8.3226089, -4.2791243, -4.0167818, 4.0093889
8: -6.6310477, -3.0642805, -6.6781731, -3.0312710, -3.5997767, 3.6138926
9: 4.0080853, 6.4742885, 3.9826174, 6.5122128, -2.5041275, 2.4916711

Time for backsubstitution: 14.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 5820
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 5820
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 494

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_B2_B1_A1_A1_A1

### Relational analysis result of IS_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8277251, upper bound: 1.7880011
time: 4.93 seconds

## Relational analysis of IS_B2_B1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 961

## Relational analysis of IS_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4597

## Relational analysis of IS_B2_B1_A1_A1_B1

### Relational analysis result of IS_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8317636, upper bound: 1.8040175
time: 4.52 seconds

## Relational analysis of IS_B2_B1_A1_A1_B2

### Relational analysis result of IS_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8318886, upper bound: 1.8053128
time: 4.68 seconds

## BFS IS instance: IS_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -9.6516857, -5.2123361, -9.7020168, -5.1795983, -4.0787992, 3.9785881
1: -17.0863495, -13.3565950, -17.0970554, -13.3739710, -3.7123785, 3.7404604
2: -8.1016855, -4.3077364, -8.1175184, -4.2952156, -3.5496011, 3.4444532
3: -13.7873831, -8.7640629, -13.7561626, -8.7565775, -5.0308056, 4.9920998
4: -3.8931284, -0.2618936, -3.8358641, -0.2915483, -3.6015801, 3.5527778
5: -13.9539986, -9.9989195, -13.9660082, -10.0149679, -3.3114119, 3.3075624
6: -15.9145517, -11.4365759, -15.9357290, -11.4394979, -4.1284628, 4.0709405
7: -8.3611193, -4.2134285, -8.3226089, -4.2791243, -4.0819950, 4.1091805
8: -6.6736207, -3.0213208, -6.6781731, -3.0312710, -3.6423497, 3.6568522
9: 3.9322443, 6.5279779, 3.9826174, 6.5122128, -2.5799685, 2.5453606

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 5820
type: A, layer: 1, pos: 4597
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 5820
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 494

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_B2_B1_A1_A2_A1

### Relational analysis result of IS_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8277252, upper bound: 1.8141908
time: 4.56 seconds

## Relational analysis of IS_B2_B1_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 961

## Relational analysis of IS_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5820

## Relational analysis of IS_B2_B1_A1_A2_B1

### Relational analysis result of IS_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8242414, upper bound: 1.8308952
time: 4.55 seconds

## Relational analysis of IS_B2_B1_A1_A2_B2

### Relational analysis result of IS_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8318834, upper bound: 1.8314887
time: 4.77 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 41.40 seconds
IS_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 41.40
Output dim: 9, lower bound: -1.8051938, upper bound: 1.8040164
IS_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 41.40
Output dim: 9, lower bound: -1.8053152, upper bound: 1.8053108
IS_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 41.40
Output dim: 9, lower bound: -1.8051938, upper bound: 1.8040165
IS_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 41.40
Output dim: 9, lower bound: -1.8053152, upper bound: 1.8053111
IS_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 41.40
Output dim: 9, lower bound: -1.8051907, upper bound: 1.8305902
IS_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 41.40
Output dim: 9, lower bound: -1.8053121, upper bound: 1.8318886
IS_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 41.40
Output dim: 9, lower bound: -1.8051907, upper bound: 1.8305898
IS_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 41.40
Output dim: 9, lower bound: -1.8053122, upper bound: 1.8318888
IS_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 41.40
Output dim: 9, lower bound: -1.8040162, upper bound: 1.8314877
IS_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 41.40
Output dim: 9, lower bound: -1.8053110, upper bound: 1.8314907
IS_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 41.40
Output dim: 9, lower bound: -1.8051896, upper bound: 1.8303202
IS_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 41.40
Output dim: 9, lower bound: -1.8053110, upper bound: 1.8314924
IS_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 41.40
Output dim: 9, lower bound: -1.8040132, upper bound: 1.8580939
IS_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 41.40
Output dim: 9, lower bound: -1.8053080, upper bound: 1.8580974
IS_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 41.40
Output dim: 9, lower bound: -1.8051867, upper bound: 1.8569266
IS_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 41.40
Output dim: 9, lower bound: -1.8053081, upper bound: 1.8580986
IS_B2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 41.40
Output dim: 9, lower bound: -1.8317636, upper bound: 1.8040175
IS_B2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 41.40
Output dim: 9, lower bound: -1.8318886, upper bound: 1.8053128
IS_B2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 41.40
Output dim: 9, lower bound: -1.8242414, upper bound: 1.8308952
IS_B2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 41.40
Output dim: 9, lower bound: -1.8318834, upper bound: 1.8314887
IS_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 41.40
Output dim: 9, lower bound: -1.8053083, upper bound: 1.8341480
IS_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 41.40
Output dim: 9, lower bound: -1.8053083, upper bound: 1.8314936
IS_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 41.40
Output dim: 9, lower bound: -1.8053082, upper bound: 1.8053082
IS_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 41.40
Output dim: 9, lower bound: -1.8053083, upper bound: 1.8314928
IS_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 41.40
Output dim: 9, lower bound: -1.8053083, upper bound: 1.8341430
IS_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 41.40
Output dim: 9, lower bound: -1.8053083, upper bound: 1.8314936
Binary search (step 0): status=Status.UNKNOWN, k_low=6, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=2.6592936515808105
rel_dist={9: [-1.860453671254131, 1.860453011622619]}

## Binary search (step 1) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5875
type: B, layer: 1, pos: 5875
type: A, layer: 1, pos: 4557
type: B, layer: 1, pos: 4557
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 5820
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4597
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 494

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5875

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7014879, upper bound: 1.6806596
time: 16.57 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7023552, upper bound: 1.7023542
time: 7.12 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 23.89 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 23.89
Output dim: 9, lower bound: -1.7014879, upper bound: 1.6806596
IS_A2, status: Status.UNKNOWN, split count: 1, time: 23.89
Output dim: 9, lower bound: -1.7023552, upper bound: 1.7023542

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -9.6517572, -5.2123337, -9.7007456, -5.1739612, -3.6819229, 3.6926937
1: -17.0863571, -13.3565741, -17.1128521, -13.3359451, -3.5621939, 3.5697231
2: -8.1017151, -4.3077283, -8.1495419, -4.2671871, -3.2640104, 3.2713065
3: -13.7874165, -8.7640533, -13.8163939, -8.7372961, -5.0501204, 5.0523405
4: -3.8931594, -0.2618493, -3.9067888, -0.2490916, -3.3719625, 3.3744388
5: -13.9540033, -9.9989052, -13.9822340, -9.9761906, -3.0261502, 3.0322227
6: -15.9145575, -11.4365091, -15.9461832, -11.4057398, -3.8387508, 3.8394623
7: -8.3611498, -4.2133913, -8.3873997, -4.1892347, -4.1169319, 4.1181641
8: -6.6736360, -3.0213003, -6.7068520, -2.9896326, -3.6840034, 3.6855516
9: 3.9322205, 6.5280085, 3.9073591, 6.5546203, -2.6223998, 2.6206493

Time for backsubstitution: 14.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4557
type: A, layer: 1, pos: 4557
type: B, layer: 1, pos: 5875
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5820
type: B, layer: 1, pos: 5820
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 4597
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 494

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4557

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6798732, upper bound: 1.6806059
time: 4.30 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7014325, upper bound: 1.6806059
time: 5.39 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -9.7212772, -5.1725125, -9.7212820, -5.1725121, -3.8806372, 3.8839037
1: -17.1234398, -13.3353319, -17.1234436, -13.3353310, -3.6267662, 3.6329155
2: -8.1695595, -4.2664623, -8.1695728, -4.2664618, -3.4623556, 3.4674077
3: -13.8282642, -8.7362051, -13.8282681, -8.7362080, -5.0920563, 5.0920630
4: -3.9074695, -0.2440007, -3.9074697, -0.2440001, -3.4036913, 3.3967166
5: -13.9941082, -9.9756899, -13.9941120, -9.9756899, -3.0984268, 3.1043143
6: -15.9595528, -11.4040365, -15.9595556, -11.4040337, -3.9622431, 3.9677229
7: -8.3881950, -4.1793032, -8.3881922, -4.1792941, -4.2059507, 4.2012720
8: -6.7207656, -2.9882154, -6.7207685, -2.9882145, -3.7325511, 3.7325530
9: 3.9066486, 6.5659370, 3.9066477, 6.5659409, -2.6592922, 2.6592894

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4557
type: B, layer: 1, pos: 4557
type: B, layer: 1, pos: 5875
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5820
type: B, layer: 1, pos: 5820
type: A, layer: 1, pos: 4597
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 494

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4557

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7023002, upper bound: 1.6807007
time: 5.06 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7023002, upper bound: 1.7022981
time: 7.03 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 26.91 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 26.91
Output dim: 9, lower bound: -1.6798732, upper bound: 1.6806059
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 26.91
Output dim: 9, lower bound: -1.7014325, upper bound: 1.6806059
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 26.91
Output dim: 9, lower bound: -1.7023002, upper bound: 1.6807007
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 26.91
Output dim: 9, lower bound: -1.7023002, upper bound: 1.7022981

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -9.6467495, -5.2136612, -9.6814775, -5.1810513, -3.6565037, 3.6742435
1: -17.0794296, -13.3644590, -17.0864697, -13.3745728, -3.5190530, 3.5347471
2: -8.0864601, -4.3120279, -8.0974989, -4.2959328, -3.2193785, 3.2197356
3: -13.7666416, -8.7671547, -13.7442751, -8.7576704, -5.0089712, 4.9771204
4: -3.8735437, -0.2703981, -3.8351841, -0.2966421, -3.3676472, 3.3017955
5: -13.9494209, -10.0093136, -13.9541349, -10.0154667, -2.9873762, 2.9968495
6: -15.9102345, -11.4469233, -15.9223557, -11.4411888, -3.7799511, 3.7841978
7: -8.3458042, -4.2395797, -8.3218660, -4.2890587, -3.9996538, 4.0060987
8: -6.6662221, -3.0330658, -6.6642599, -3.0326791, -3.6335430, 3.6311941
9: 3.9445324, 6.5131068, 3.9833031, 6.5008965, -2.5563641, 2.5298038

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4557
type: B, layer: 1, pos: 5875
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 5820
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 5820
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 494

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4557

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6798731, upper bound: 1.6590683
time: 4.39 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6798731, upper bound: 1.6806059
time: 4.45 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -9.6517572, -5.2123337, -9.7007294, -5.1739626, -3.6806755, 3.6928101
1: -17.0863571, -13.3565741, -17.1128464, -13.3359604, -3.5621829, 3.5681839
2: -8.1017151, -4.3077283, -8.1495161, -4.2671928, -3.2640047, 3.2464910
3: -13.7874165, -8.7640533, -13.8163595, -8.7373018, -5.0501146, 5.0523062
4: -3.8931594, -0.2618493, -3.9067566, -0.2491059, -3.3719459, 3.3514051
5: -13.9540033, -9.9989052, -13.9822283, -9.9761963, -3.0182910, 3.0322151
6: -15.9145575, -11.4365091, -15.9461765, -11.4057550, -3.8487902, 3.8320050
7: -8.3611498, -4.2133913, -8.3873749, -4.1892715, -4.1078596, 4.1181402
8: -6.6736360, -3.0213003, -6.7068405, -2.9896517, -3.6839843, 3.6855402
9: 3.9322205, 6.5280085, 3.9073792, 6.5545888, -2.6223683, 2.6206293

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5875
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 4557
type: B, layer: 1, pos: 5820
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 5820
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 494

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5875

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6807534, upper bound: 1.6806055
time: 17.95 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6807534, upper bound: 1.6806056
time: 5.04 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -9.7020168, -5.1795983, -9.7162790, -5.1738358, -3.8622494, 3.8567832
1: -17.0970554, -13.3739710, -17.1165237, -13.3432350, -3.5917702, 3.5897622
2: -8.1175184, -4.2952156, -8.1543150, -4.2707682, -3.4107218, 3.4170918
3: -13.7561626, -8.7565775, -13.8075027, -8.7393017, -5.0168610, 5.0509253
4: -3.8358641, -0.2915483, -3.8878522, -0.2525259, -3.3310699, 3.3925414
5: -13.9660082, -10.0149679, -13.9895287, -9.9861002, -3.0629849, 3.0654216
6: -15.9357290, -11.4394979, -15.9552269, -11.4144697, -3.9068995, 3.9088411
7: -8.3226089, -4.2791243, -8.3728085, -4.2054777, -4.0939398, 4.0838585
8: -6.6781731, -3.0312710, -6.7133522, -2.9999990, -3.6781740, 3.6820812
9: 3.9826174, 6.5122128, 3.9189920, 6.5510387, -2.5684214, 2.5932207

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5875
type: B, layer: 1, pos: 4557
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5820
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 5820
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 494

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5875

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6806057, upper bound: 1.6798719
time: 5.04 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6806057, upper bound: 1.6807016
time: 4.88 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -9.7212629, -5.1725130, -9.7212820, -5.1725121, -3.8807640, 3.8823166
1: -17.1234303, -13.3353443, -17.1234436, -13.3353310, -3.6252270, 3.6329031
2: -8.1695328, -4.2664700, -8.1695728, -4.2664618, -3.4375401, 3.4634485
3: -13.8282261, -8.7362127, -13.8282681, -8.7362080, -5.0920181, 5.0920553
4: -3.9074385, -0.2440178, -3.9074697, -0.2440001, -3.3806581, 3.3966990
5: -13.9940996, -9.9756966, -13.9941120, -9.9756899, -3.0984201, 3.0964546
6: -15.9595490, -11.4040546, -15.9595556, -11.4040337, -3.9547863, 3.9777594
7: -8.3881712, -4.1793394, -8.3881922, -4.1792941, -4.2059288, 4.1921997
8: -6.7207527, -2.9882340, -6.7207685, -2.9882145, -3.7325382, 3.7325344
9: 3.9066682, 6.5659094, 3.9066477, 6.5659409, -2.6592727, 2.6592617

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5875
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 4557
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4597
type: B, layer: 1, pos: 5820
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 494

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5875

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6806057, upper bound: 1.7014316
time: 7.76 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6806057, upper bound: 1.7014327
time: 4.62 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 27.18 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 27.18
Output dim: 9, lower bound: -1.6798731, upper bound: 1.6590683
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 27.18
Output dim: 9, lower bound: -1.6798731, upper bound: 1.6806059
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 27.18
Output dim: 9, lower bound: -1.6807534, upper bound: 1.6806055
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 27.18
Output dim: 9, lower bound: -1.6807534, upper bound: 1.6806056
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 27.18
Output dim: 9, lower bound: -1.6806057, upper bound: 1.6798719
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 27.18
Output dim: 9, lower bound: -1.6806057, upper bound: 1.6807016
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 27.18
Output dim: 9, lower bound: -1.6806057, upper bound: 1.7014316
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 27.18
Output dim: 9, lower bound: -1.6806057, upper bound: 1.7014327

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -9.6324787, -5.2194176, -9.6814775, -5.1810513, -3.6420221, 3.6528687
1: -17.0599537, -13.3951693, -17.0864697, -13.3745728, -3.4979353, 3.5055313
2: -8.0496616, -4.3364553, -8.0974989, -4.2959328, -3.1872807, 3.1946096
3: -13.7152596, -8.7844372, -13.7442751, -8.7576704, -4.9575891, 4.9598379
4: -3.8215551, -0.3094628, -3.8351841, -0.2966421, -3.3246737, 3.3270659
5: -13.9259109, -10.0381699, -13.9541349, -10.0154667, -2.9646606, 2.9707351
6: -15.8907461, -11.4718838, -15.9223557, -11.4411888, -3.7533178, 3.7540712
7: -8.2959061, -4.3132200, -8.3218660, -4.2890587, -3.9363251, 3.9373565
8: -6.6310477, -3.0642805, -6.6642599, -3.0326791, -3.5983686, 3.5999794
9: 4.0080853, 6.4742885, 3.9833031, 6.5008965, -2.4928112, 2.4909854

Time for backsubstitution: 14.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5875
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5820
type: B, layer: 1, pos: 5820
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 4597
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 494

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5875

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6592171, upper bound: 1.6590727
time: 4.88 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6592171, upper bound: 1.6590729
time: 5.20 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -9.6516857, -5.2123361, -9.6814775, -5.1810513, -3.6631508, 3.6742067
1: -17.0863495, -13.3565950, -17.0864697, -13.3745728, -3.5270667, 3.5408926
2: -8.1016855, -4.3077364, -8.0974989, -4.2959328, -3.2340221, 3.2242517
3: -13.7873831, -8.7640629, -13.7442751, -8.7576704, -5.0297127, 4.9802122
4: -3.8931284, -0.2618936, -3.8351841, -0.2966421, -3.3905106, 3.3103433
5: -13.9539986, -9.9989195, -13.9541349, -10.0154667, -2.9909925, 3.0074534
6: -15.9145517, -11.4365759, -15.9223557, -11.4411888, -3.7829762, 3.7932868
7: -8.3611193, -4.2134285, -8.3218660, -4.2890587, -4.0140047, 4.0311155
8: -6.6736207, -3.0213208, -6.6642599, -3.0326791, -3.6409416, 3.6429391
9: 3.9322443, 6.5279779, 3.9833031, 6.5008965, -2.5686522, 2.5446749

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5875
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 5820
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 5820
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 494

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5875

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6592171, upper bound: 1.6806059
time: 4.46 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6592170, upper bound: 1.6806059
time: 5.61 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -9.6517572, -5.2123337, -9.6517410, -5.2123356, -3.6423159, 3.6436610
1: -17.0863571, -13.3565741, -17.0863514, -13.3565893, -3.5429149, 3.5413876
2: -8.1017151, -4.3077283, -8.1016903, -4.3077345, -3.2234802, 3.1986675
3: -13.7874165, -8.7640533, -13.7873850, -8.7640591, -5.0233574, 5.0233316
4: -3.8931594, -0.2618493, -3.8931270, -0.2618673, -3.3601117, 3.3370967
5: -13.9540033, -9.9989052, -13.9539986, -9.9989147, -2.9958425, 3.0036960
6: -15.9145575, -11.4365091, -15.9145527, -11.4365282, -3.8179941, 3.8004756
7: -8.3611498, -4.2133913, -8.3611259, -4.2134304, -4.0833063, 4.0923557
8: -6.6736360, -3.0213003, -6.6736231, -3.0213170, -3.6523190, 3.6523228
9: 3.9322205, 6.5280085, 3.9322391, 6.5279784, -2.5957580, 2.5957694

Time for backsubstitution: 14.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 4557
type: B, layer: 1, pos: 5820
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 5820
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 494

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 961

## Relational analysis of IS_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4557

## Relational analysis of IS_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6592171, upper bound: 1.6590683
time: 4.70 seconds

## Relational analysis of IS_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6592170, upper bound: 1.6806058
time: 5.07 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -9.6517572, -5.2123337, -9.7212362, -5.1737866, -3.6779175, 3.7134657
1: -17.0863571, -13.3565741, -17.1233864, -13.3356800, -3.5624313, 3.5792046
2: -8.1017151, -4.3077283, -8.1695175, -4.2671132, -3.2476583, 3.2652316
3: -13.7874165, -8.7640533, -13.8281803, -8.7366867, -5.0507298, 5.0641270
4: -3.8931594, -0.2618493, -3.9072380, -0.2440574, -3.3778763, 3.3512096
5: -13.9540033, -9.9989052, -13.9940891, -9.9761477, -3.0177732, 3.0442533
6: -15.9145575, -11.4365091, -15.9595337, -11.4049778, -3.8496923, 3.8453884
7: -8.3611498, -4.2133913, -8.3877459, -4.1793680, -4.1180582, 4.1185131
8: -6.6736360, -3.0213003, -6.7207437, -2.9891171, -3.6845188, 3.6994433
9: 3.9322205, 6.5280085, 3.9070263, 6.5659070, -2.6336865, 2.6209822

Time for backsubstitution: 14.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 4557
type: B, layer: 1, pos: 5820
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 5820
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 494

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 961

## Relational analysis of IS_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4557

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6592171, upper bound: 1.6590682
time: 4.68 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6592170, upper bound: 1.6806059
time: 5.23 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: -9.7020168, -5.1795983, -9.6467495, -5.2136612, -3.6950684, 3.7865973
1: -17.0970554, -13.3739710, -17.0794296, -13.3644590, -3.5458002, 3.5518923
2: -8.1175184, -4.2952156, -8.0864601, -4.3120279, -3.2383337, 3.3444517
3: -13.7561626, -8.7565775, -13.7666416, -8.7671547, -4.9890079, 5.0100641
4: -3.8358641, -0.2915483, -3.8735437, -0.2703981, -3.3124542, 3.3740921
5: -13.9660082, -10.0149679, -13.9494209, -10.0093136, -3.0088916, 3.0247159
6: -15.9357290, -11.4394979, -15.9102345, -11.4469233, -3.7976189, 3.8638768
7: -8.3226089, -4.2791243, -8.3458042, -4.2395797, -4.0528259, 4.0098896
8: -6.6781731, -3.0312710, -6.6662221, -3.0330658, -3.6451073, 3.6349511
9: 3.9826174, 6.5122128, 3.9445324, 6.5131068, -2.5304894, 2.5676804

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4557
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5820
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 5820
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 494

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4557

## Relational analysis of IS_A2_A1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6590728, upper bound: 1.6798723
time: 4.87 seconds

## Relational analysis of IS_A2_A1_B1_B2

### Relational analysis result of IS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6590727, upper bound: 1.6798723
time: 5.50 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: -9.7020168, -5.1795983, -9.7162743, -5.1738381, -3.8622503, 3.8540401
1: -17.0970554, -13.3739710, -17.1165142, -13.3432341, -3.5917692, 3.5836134
2: -8.1175184, -4.2952156, -8.1543026, -4.2707682, -3.4157686, 3.4170916
3: -13.7561626, -8.7565775, -13.8075008, -8.7393017, -5.0168610, 5.0509233
4: -3.8358641, -0.2915483, -3.8878503, -0.2525269, -3.3240957, 3.3925405
5: -13.9660082, -10.0149679, -13.9895229, -9.9860983, -3.0629845, 3.0595345
6: -15.9357290, -11.4394979, -15.9552279, -11.4144745, -3.9068995, 3.9033599
7: -8.3226089, -4.2791243, -8.3728075, -4.2054820, -4.0939341, 4.0885296
8: -6.6781731, -3.0312710, -6.7133470, -3.0000010, -3.6781721, 3.6820760
9: 3.9826174, 6.5122128, 3.9189930, 6.5510359, -2.5684185, 2.5932198

Time for backsubstitution: 14.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4557
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 5820
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 5820
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 494

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4557

## Relational analysis of IS_A2_A1_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6590727, upper bound: 1.6807021
time: 4.66 seconds

## Relational analysis of IS_A2_A1_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6590727, upper bound: 1.6807030
time: 5.12 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -9.7212629, -5.1725130, -9.6517572, -5.2123337, -3.7135820, 3.8122222
1: -17.1234303, -13.3353443, -17.0863571, -13.3565741, -3.5792317, 3.5950394
2: -8.1695328, -4.2664700, -8.1017151, -4.3077283, -3.2652478, 3.3908145
3: -13.8282261, -8.7362127, -13.7874165, -8.7640533, -5.0641727, 5.0512037
4: -3.9074385, -0.2440178, -3.8931594, -0.2618493, -3.3620491, 3.3784175
5: -13.9940996, -9.9756966, -13.9540033, -9.9989052, -3.0442605, 3.0557451
6: -15.9595490, -11.4040546, -15.9145575, -11.4365091, -3.8454237, 3.9328089
7: -8.3881712, -4.1793394, -8.3611498, -4.2133913, -4.1649170, 4.1180887
8: -6.7207527, -2.9882340, -6.6736360, -3.0213003, -3.6994524, 3.6854019
9: 3.9066682, 6.5659094, 3.9322205, 6.5280085, -2.6213403, 2.6336889

Time for backsubstitution: 15.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 4557
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4597
type: B, layer: 1, pos: 5820
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 494

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 961

## Relational analysis of IS_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6634021, upper bound: 1.6958665
time: 5.50 seconds

## Relational analysis of IS_A2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4557

## Relational analysis of IS_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6590682, upper bound: 1.7014307
time: 4.75 seconds

## Relational analysis of IS_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6590682, upper bound: 1.7014322
time: 9.49 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -9.7212629, -5.1725130, -9.7212772, -5.1725125, -3.8807640, 3.8793869
1: -17.1234303, -13.3353443, -17.1234398, -13.3353319, -3.6252270, 3.6267538
2: -8.1695328, -4.2664700, -8.1695595, -4.2664623, -3.4425850, 3.4634480
3: -13.8282261, -8.7362127, -13.8282642, -8.7362051, -5.0920210, 5.0920515
4: -3.9074385, -0.2440178, -3.9074695, -0.2440007, -3.3736830, 3.3966990
5: -13.9940996, -9.9756966, -13.9941082, -9.9756899, -3.0984201, 3.0905671
6: -15.9595490, -11.4040546, -15.9595528, -11.4040365, -3.9547853, 3.9722795
7: -8.3881712, -4.1793394, -8.3881950, -4.1793032, -4.2059212, 4.1968727
8: -6.7207527, -2.9882340, -6.7207656, -2.9882154, -3.7325373, 3.7325315
9: 3.9066682, 6.5659094, 3.9066486, 6.5659370, -2.6592689, 2.6592607

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 4557
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4597
type: B, layer: 1, pos: 5820
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 494

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 961

## Relational analysis of IS_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6634021, upper bound: 1.7021889
time: 5.86 seconds

## Relational analysis of IS_A2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4557

## Relational analysis of IS_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6590682, upper bound: 1.7023000
time: 4.36 seconds

## Relational analysis of IS_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6590682, upper bound: 1.7023012
time: 4.50 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 42.22 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 42.22
Output dim: 9, lower bound: -1.6592171, upper bound: 1.6590727
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 42.22
Output dim: 9, lower bound: -1.6592171, upper bound: 1.6590729
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 42.22
Output dim: 9, lower bound: -1.6592171, upper bound: 1.6806059
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 42.22
Output dim: 9, lower bound: -1.6592170, upper bound: 1.6806059
IS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 42.22
Output dim: 9, lower bound: -1.6592171, upper bound: 1.6590683
IS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 42.22
Output dim: 9, lower bound: -1.6592170, upper bound: 1.6806058
IS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 42.22
Output dim: 9, lower bound: -1.6592171, upper bound: 1.6590682
IS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 42.22
Output dim: 9, lower bound: -1.6592170, upper bound: 1.6806059
IS_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 42.22
Output dim: 9, lower bound: -1.6590728, upper bound: 1.6798723
IS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 42.22
Output dim: 9, lower bound: -1.6590727, upper bound: 1.6798723
IS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 42.22
Output dim: 9, lower bound: -1.6590727, upper bound: 1.6807021
IS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 42.22
Output dim: 9, lower bound: -1.6590727, upper bound: 1.6807030
IS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 42.22
Output dim: 9, lower bound: -1.6590682, upper bound: 1.7014307
IS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 42.22
Output dim: 9, lower bound: -1.6590682, upper bound: 1.7014322
IS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 42.22
Output dim: 9, lower bound: -1.6590682, upper bound: 1.7023000
IS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 42.22
Output dim: 9, lower bound: -1.6590682, upper bound: 1.7023012

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -9.6324787, -5.2194176, -9.6324787, -5.2194176, -3.6036730, 3.6036735
1: -17.0599537, -13.3951693, -17.0599537, -13.3951693, -3.4787235, 3.4787235
2: -8.0496616, -4.3364553, -8.0496616, -4.3364553, -3.1467781, 3.1467776
3: -13.7152596, -8.7844372, -13.7152596, -8.7844372, -4.9308224, 4.9308224
4: -3.8215551, -0.3094628, -3.8215551, -0.3094628, -3.3127584, 3.3127589
5: -13.9259109, -10.0381699, -13.9259109, -10.0381699, -2.9422331, 2.9422331
6: -15.8907461, -11.4718838, -15.8907461, -11.4718838, -3.7225609, 3.7225614
7: -8.2959061, -4.3132200, -8.2959061, -4.3132200, -3.9116468, 3.9116468
8: -6.6310477, -3.0642805, -6.6310477, -3.0642805, -3.5667672, 3.5667672
9: 4.0080853, 6.4742885, 4.0080853, 6.4742885, -2.4662032, 2.4662032

Time for backsubstitution: 14.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 5820
type: B, layer: 1, pos: 5820
type: A, layer: 1, pos: 4597
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 494

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 961

## Relational analysis of IS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5820

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6584751, upper bound: 1.6527296
time: 12.10 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6592161, upper bound: 1.6590671
time: 4.37 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -9.6324787, -5.2194176, -9.7019901, -5.1808753, -3.6380949, 3.6735878
1: -17.0599537, -13.3951693, -17.0970135, -13.3743057, -3.4981661, 3.5165567
2: -8.0496616, -4.3364553, -8.1175013, -4.2958589, -3.1794496, 3.2128496
3: -13.7152596, -8.7844372, -13.7561131, -8.7570534, -4.9582062, 4.9716759
4: -3.8215551, -0.3094628, -3.8356650, -0.2915885, -3.3305788, 3.3268676
5: -13.9259109, -10.0381699, -13.9660034, -10.0154209, -2.9641204, 2.9827695
6: -15.8907461, -11.4718838, -15.9357185, -11.4404163, -3.7541981, 3.7674565
7: -8.2959061, -4.3132200, -8.3221769, -4.2791514, -3.9465761, 3.9376678
8: -6.6310477, -3.0642805, -6.6781597, -3.0321503, -3.5988975, 3.6138792
9: 4.0080853, 6.4742885, 3.9829741, 6.5122108, -2.5041256, 2.4913144

Time for backsubstitution: 14.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 5820
type: B, layer: 1, pos: 5820
type: A, layer: 1, pos: 4597
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 494

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 961

## Relational analysis of IS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5820

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6584751, upper bound: 1.6527292
time: 6.79 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6592161, upper bound: 1.6590671
time: 4.87 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9.6516857, -5.2123361, -9.6324787, -5.2194176, -3.6248026, 3.6250324
1: -17.0863495, -13.3565950, -17.0599537, -13.3951693, -3.5078540, 3.5140843
2: -8.1016855, -4.3077364, -8.0496616, -4.3364553, -3.1935158, 3.1764197
3: -13.7873831, -8.7640629, -13.7152596, -8.7844372, -5.0029459, 4.9511967
4: -3.8931284, -0.2618936, -3.8215551, -0.3094628, -3.3790331, 3.2960415
5: -13.9539986, -9.9989195, -13.9259109, -10.0381699, -2.9685650, 2.9789519
6: -15.9145517, -11.4365759, -15.8907461, -11.4718838, -3.7522182, 3.7617764
7: -8.3611193, -4.2134285, -8.2959061, -4.3132200, -3.9894342, 4.0052924
8: -6.6736207, -3.0213208, -6.6310477, -3.0642805, -3.6093402, 3.6097269
9: 3.9322443, 6.5279779, 4.0080853, 6.4742885, -2.5420442, 2.5198927

Time for backsubstitution: 14.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 5820
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 5820
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 494

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 961

## Relational analysis of IS_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5820

## Relational analysis of IS_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6528739, upper bound: 1.6801363
time: 4.71 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6592113, upper bound: 1.6805981
time: 4.76 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.6516857, -5.2123361, -9.7019901, -5.1808753, -3.6566086, 3.6949153
1: -17.0863495, -13.3565950, -17.0970135, -13.3743057, -3.5272965, 3.5519180
2: -8.1016855, -4.3077364, -8.1175013, -4.2958589, -3.2064409, 3.2339439
3: -13.7873831, -8.7640629, -13.7561131, -8.7570534, -5.0303297, 4.9920502
4: -3.8931284, -0.2618936, -3.8356650, -0.2915885, -3.3906527, 3.3101420
5: -13.9539986, -9.9989195, -13.9660034, -10.0154209, -2.9904523, 3.0164118
6: -15.9145517, -11.4365759, -15.9357185, -11.4404163, -3.7838554, 3.8066726
7: -8.3611193, -4.2134285, -8.3221769, -4.2791514, -4.0242100, 4.0136304
8: -6.6736207, -3.0213208, -6.6781597, -3.0321503, -3.6414704, 3.6568389
9: 3.9322443, 6.5279779, 3.9829741, 6.5122108, -2.5799665, 2.5450039

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 5820
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 5820
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 494

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 961

## Relational analysis of IS_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5820

## Relational analysis of IS_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6528739, upper bound: 1.6801356
time: 4.69 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6592112, upper bound: 1.6805978
time: 5.96 seconds

## BFS IS instance: IS_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -9.6324787, -5.2194176, -9.6517410, -5.2123356, -3.6250324, 3.6241865
1: -17.0599537, -13.3951693, -17.0863514, -13.3565893, -3.5140872, 3.5075407
2: -8.0496616, -4.3364553, -8.1016903, -4.3077345, -3.1762857, 3.1935191
3: -13.7152596, -8.7844372, -13.7873850, -8.7640591, -4.9512005, 5.0029478
4: -3.8215551, -0.3094628, -3.8931270, -0.2618673, -3.2932935, 3.3790317
5: -13.9259109, -10.0381699, -13.9539986, -9.9989147, -2.9773564, 2.9685650
6: -15.8907461, -11.4718838, -15.9145527, -11.4365282, -3.7708540, 3.7522192
7: -8.2959061, -4.3132200, -8.3611259, -4.2134304, -4.0052967, 3.9915333
8: -6.6310477, -3.0642805, -6.6736231, -3.0213170, -3.6097307, 3.6093426
9: 4.0080853, 6.4742885, 3.9322391, 6.5279784, -2.5198932, 2.5420494

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 5820
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 5820
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 494

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 961

## Relational analysis of IS_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5820

## Relational analysis of IS_A1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6584707, upper bound: 1.6528735
time: 10.84 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6592116, upper bound: 1.6592134
time: 4.39 seconds

## BFS IS instance: IS_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -9.6517410, -5.2123356, -9.6517410, -5.2123356, -3.6436548, 3.6436543
1: -17.0863514, -13.3565893, -17.0863514, -13.3565893, -3.5413761, 3.5413756
2: -8.1016903, -4.3077345, -8.1016903, -4.3077345, -3.1986609, 3.1986609
3: -13.7873850, -8.7640591, -13.7873850, -8.7640591, -5.0233259, 5.0233259
4: -3.8931270, -0.2618673, -3.8931270, -0.2618673, -3.3370781, 3.3370786
5: -13.9539986, -9.9989147, -13.9539986, -9.9989147, -2.9958358, 2.9958363
6: -15.9145527, -11.4365282, -15.9145527, -11.4365282, -3.8179712, 3.8179717
7: -8.3611259, -4.2134304, -8.3611259, -4.2134304, -4.0832825, 4.0832825
8: -6.6736231, -3.0213170, -6.6736231, -3.0213170, -3.6523061, 3.6523061
9: 3.9322391, 6.5279784, 3.9322391, 6.5279784, -2.5957394, 2.5957394

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 5820
type: B, layer: 1, pos: 5820
type: A, layer: 1, pos: 4597
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 494

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 961

## Relational analysis of IS_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5820

## Relational analysis of IS_A1_B2_B1_A2_A1

### Relational analysis result of IS_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6584707, upper bound: 1.6528734
time: 6.61 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2

### Relational analysis result of IS_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6592115, upper bound: 1.6807471
time: 5.78 seconds

## BFS IS instance: IS_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -9.6324787, -5.2194176, -9.7212362, -5.1737866, -3.6606913, 3.6916714
1: -17.0599537, -13.3951693, -17.1233864, -13.3356800, -3.5336027, 3.5453596
2: -8.0496616, -4.3364553, -8.1695175, -4.2671132, -3.2004309, 3.2398510
3: -13.7152596, -8.7844372, -13.8281803, -8.7366867, -4.9785728, 5.0437431
4: -3.8215551, -0.3094628, -3.9072380, -0.2440574, -3.3110590, 3.3847864
5: -13.9259109, -10.0381699, -13.9940891, -9.9761477, -2.9943929, 3.0091238
6: -15.8907461, -11.4718838, -15.9595337, -11.4049778, -3.8025618, 3.7971354
7: -8.2959061, -4.3132200, -8.3877459, -4.1793680, -4.0253735, 4.0176916
8: -6.6310477, -3.0642805, -6.7207437, -2.9891171, -3.6419306, 3.6564631
9: 4.0080853, 6.4742885, 3.9070263, 6.5659070, -2.5578218, 2.5672622

Time for backsubstitution: 15.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 5820
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 5820
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 494

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A1_B2_B2_A1_A1

### Relational analysis result of IS_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6742964, upper bound: 1.6418500
time: 5.14 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 961

## Relational analysis of IS_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5820

## Relational analysis of IS_A1_B2_B2_A1_A1

### Relational analysis result of IS_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6791190, upper bound: 1.6527270
time: 8.67 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2

### Relational analysis result of IS_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6798665, upper bound: 1.6590625
time: 5.02 seconds

## BFS IS instance: IS_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -9.6517410, -5.2123356, -9.7212362, -5.1737866, -3.6794453, 3.7134595
1: -17.0863514, -13.3565893, -17.1233864, -13.3356800, -3.5608912, 3.5791926
2: -8.1016903, -4.3077345, -8.1695175, -4.2671132, -3.2314010, 3.2647867
3: -13.7873850, -8.7640591, -13.8281803, -8.7366867, -5.0506983, 5.0641212
4: -3.8931270, -0.2618673, -3.9072380, -0.2440574, -3.3548431, 3.3511910
5: -13.9539986, -9.9989147, -13.9940891, -9.9761477, -3.0177665, 3.0363946
6: -15.9145527, -11.4365282, -15.9595337, -11.4049778, -3.8496695, 3.8628678
7: -8.3611259, -4.2134304, -8.3877459, -4.1793680, -4.1180334, 4.1094418
8: -6.6736231, -3.0213170, -6.7207437, -2.9891171, -3.6845059, 3.6994267
9: 3.9322391, 6.5279784, 3.9070263, 6.5659070, -2.6336679, 2.6209521

Time for backsubstitution: 15.35 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=6, k_high=8, k_mid=7, eps_mid=0.0273438, abs_max=2.6592936515808105
rel_dist={9: [-1.702359651903449, 1.7023598330847598]}

## Binary search (step 2) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5875
type: B, layer: 1, pos: 5875
type: A, layer: 1, pos: 4557
type: B, layer: 1, pos: 4557
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 5820
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 494

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 5875

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6174309, upper bound: 1.5990262
time: 5.72 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6176209, upper bound: 1.6176201
time: 5.92 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 11.89 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 11.89
Output dim: 9, lower bound: -1.6174309, upper bound: 1.5990262
IS_A2, status: Status.UNKNOWN, split count: 1, time: 11.89
Output dim: 9, lower bound: -1.6176209, upper bound: 1.6176201

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -9.6517572, -5.2123337, -9.6966181, -5.1739926, -3.5400295, 3.5466924
1: -17.0863571, -13.3565741, -17.1107311, -13.3360004, -3.4299026, 3.4352641
2: -8.1017151, -4.3077283, -8.1455183, -4.2672038, -3.1627254, 3.1660075
3: -13.7874165, -8.7640533, -13.8140144, -8.7374201, -4.9471741, 4.9471531
4: -3.8931594, -0.2618493, -3.9066923, -0.2501051, -3.2546310, 3.2581706
5: -13.9540033, -9.9989052, -13.9798508, -9.9762020, -2.8822432, 2.8859043
6: -15.9145575, -11.4365091, -15.9434948, -11.4058952, -3.7067118, 3.7049065
7: -8.3611498, -4.2133913, -8.3873243, -4.1912270, -4.0025139, 4.0057220
8: -6.6736360, -3.0213003, -6.7040544, -2.9897418, -3.6422186, 3.6403813
9: 3.9322205, 6.5280085, 3.9074297, 6.5523429, -2.6201224, 2.6205788

Time for backsubstitution: 15.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4557
type: A, layer: 1, pos: 4557
type: B, layer: 1, pos: 5875
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 5820
type: B, layer: 1, pos: 5820
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 494

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 4557

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5981700, upper bound: 1.5989753
time: 6.51 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6173802, upper bound: 1.5989754
time: 8.74 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -9.7212772, -5.1725125, -9.7212820, -5.1725101, -3.7396374, 3.7411366
1: -17.1234398, -13.3353319, -17.1234455, -13.3353291, -3.4955902, 3.5020990
2: -8.1695595, -4.2664623, -8.1695709, -4.2664580, -3.3664551, 3.3714600
3: -13.8282642, -8.7362051, -13.8282661, -8.7362070, -5.0470066, 5.0566254
4: -3.9074695, -0.2440007, -3.9074697, -0.2440003, -3.2880268, 3.2806420
5: -13.9941082, -9.9756899, -13.9941149, -9.9756889, -2.9567566, 2.9629893
6: -15.9595528, -11.4040365, -15.9595547, -11.4040356, -3.8314428, 3.8372436
7: -8.3881950, -4.1793032, -8.3881950, -4.1792946, -4.0955315, 4.0908966
8: -6.7207656, -2.9882154, -6.7207718, -2.9882140, -3.7325516, 3.7325563
9: 3.9066486, 6.5659370, 3.9066477, 6.5659399, -2.6592913, 2.6592894

Time for backsubstitution: 15.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4557
type: B, layer: 1, pos: 4557
type: B, layer: 1, pos: 5875
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5820
type: B, layer: 1, pos: 5820
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 494

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 4557

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6175691, upper bound: 1.5982988
time: 6.49 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6175706, upper bound: 1.6175695
time: 5.88 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 27.92 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 27.92
Output dim: 9, lower bound: -1.5981700, upper bound: 1.5989753
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 27.92
Output dim: 9, lower bound: -1.6173802, upper bound: 1.5989754
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 27.92
Output dim: 9, lower bound: -1.6175691, upper bound: 1.5982988
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 27.92
Output dim: 9, lower bound: -1.6175706, upper bound: 1.6175695

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -9.6517572, -5.2123337, -9.6966047, -5.1739979, -3.5387802, 3.5465384
1: -17.0863571, -13.3565741, -17.1107254, -13.3360157, -3.4298897, 3.4336343
2: -8.1017151, -4.3077283, -8.1454897, -4.2672081, -3.1627178, 3.1397371
3: -13.7874165, -8.7640533, -13.8139782, -8.7374249, -4.9471693, 4.9454126
4: -3.8931594, -0.2618493, -3.9066582, -0.2501222, -3.2546134, 3.2337871
5: -13.9540033, -9.9989052, -13.9798422, -9.9762096, -2.8739214, 2.8858981
6: -15.9145575, -11.4365091, -15.9434900, -11.4059124, -3.7153006, 3.6974516
7: -8.3611498, -4.2133913, -8.3872986, -4.1912642, -3.9929104, 4.0057020
8: -6.6736360, -3.0213003, -6.7040415, -2.9897594, -3.6265154, 3.6403651
9: 3.9322205, 6.5280085, 3.9074507, 6.5523119, -2.6200914, 2.6205578

Time for backsubstitution: 15.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5875
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 4557
type: B, layer: 1, pos: 5820
type: A, layer: 1, pos: 5820
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 494

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 5875

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5997001, upper bound: 1.5989752
time: 8.41 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5997001, upper bound: 1.5989746
time: 9.78 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -9.7020168, -5.1795983, -9.7153854, -5.1740875, -3.7210379, 3.7120590
1: -17.0970554, -13.3739710, -17.1152916, -13.3447151, -3.4594059, 3.4575071
2: -8.1175184, -4.2952156, -8.1515369, -4.2715621, -3.3151798, 3.3160148
3: -13.7561626, -8.7565775, -13.8037415, -8.7398663, -5.0096273, 5.0090656
4: -3.8358641, -0.2915483, -3.8842888, -0.2540413, -3.2135568, 3.2695680
5: -13.9660082, -10.0149679, -13.9886637, -9.9879847, -2.9192481, 2.9232049
6: -15.9357290, -11.4394979, -15.9544220, -11.4163408, -3.7741642, 3.7768097
7: -8.3226089, -4.2791243, -8.3699293, -4.2102227, -3.9755688, 3.9709253
8: -6.6781731, -3.0312710, -6.7119708, -3.0021420, -3.6760311, 3.6806998
9: 3.9826174, 6.5122128, 3.9212894, 6.5483236, -2.5657063, 2.5909233

Time for backsubstitution: 15.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5875
type: B, layer: 1, pos: 4557
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 5820
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 4597
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 494

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 5875

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5989752, upper bound: 1.5981686
time: 8.85 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5989751, upper bound: 1.5982999
time: 9.23 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -9.7212629, -5.1725130, -9.7212820, -5.1725101, -3.7394953, 3.7395492
1: -17.1234303, -13.3353443, -17.1234455, -13.3353291, -3.4939623, 3.5020876
2: -8.1695328, -4.2664700, -8.1695709, -4.2664580, -3.3401804, 3.3651075
3: -13.8282261, -8.7362127, -13.8282661, -8.7362070, -5.0452662, 5.0566177
4: -3.9074385, -0.2440178, -3.9074697, -0.2440003, -3.2636433, 3.2806249
5: -13.9940996, -9.9756966, -13.9941149, -9.9756889, -2.9567490, 2.9546666
6: -15.9595490, -11.4040546, -15.9595547, -11.4040356, -3.8239841, 3.8458319
7: -8.3881712, -4.1793394, -8.3881950, -4.1792946, -4.0955095, 4.0812960
8: -6.7207527, -2.9882340, -6.7207718, -2.9882140, -3.7325387, 3.7325377
9: 3.9066682, 6.5659094, 3.9066477, 6.5659399, -2.6592717, 2.6592617

Time for backsubstitution: 15.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5875
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 4557
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4597
type: B, layer: 1, pos: 5820
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 906

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 5875

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5989751, upper bound: 1.6173792
time: 8.96 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5989751, upper bound: 1.6175689
time: 9.05 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 33.58 seconds
IS_A1_B2_B1, status: Status.VERIFIED, split count: 3, time: 33.58
Output dim: 9, lower bound: -1.5997001, upper bound: 1.5989752
IS_A1_B2_B2, status: Status.VERIFIED, split count: 3, time: 33.58
Output dim: 9, lower bound: -1.5997001, upper bound: 1.5989746
IS_A2_A1_B1, status: Status.VERIFIED, split count: 3, time: 33.58
Output dim: 9, lower bound: -1.5989752, upper bound: 1.5981686
IS_A2_A1_B2, status: Status.VERIFIED, split count: 3, time: 33.58
Output dim: 9, lower bound: -1.5989751, upper bound: 1.5982999
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 33.58
Output dim: 9, lower bound: -1.5989751, upper bound: 1.6173792
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 33.58
Output dim: 9, lower bound: -1.5989751, upper bound: 1.6175689

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -9.7212629, -5.1725130, -9.6517572, -5.2123337, -3.5665030, 3.6694551
1: -17.1234303, -13.3353443, -17.0863571, -13.3565741, -3.4469008, 3.4642243
2: -8.1695328, -4.2664700, -8.1017151, -4.3077283, -3.1562371, 3.2926903
3: -13.8282261, -8.7362127, -13.7874165, -8.7640533, -4.9597778, 5.0155935
4: -3.9074385, -0.2440178, -3.8931594, -0.2618493, -3.2450342, 3.2622786
5: -13.9940996, -9.9756966, -13.9540033, -9.9989052, -2.9001951, 2.9139581
6: -15.9595490, -11.4040546, -15.9145575, -11.4365091, -3.7135649, 3.8008814
7: -8.3881712, -4.1793394, -8.3611498, -4.2133913, -4.0546989, 4.0051928
8: -6.7207527, -2.9882340, -6.6736360, -3.0213003, -3.6581869, 3.6854019
9: 3.9066682, 6.5659094, 3.9322205, 6.5280085, -2.6213403, 2.6336889

Time for backsubstitution: 15.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 4557
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4597
type: B, layer: 1, pos: 5820
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 494

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 961

## Relational analysis of IS_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5823409, upper bound: 1.6111430
time: 7.45 seconds

## Relational analysis of IS_A2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4557

## Relational analysis of IS_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5797849, upper bound: 1.6173785
time: 5.16 seconds

## Relational analysis of IS_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5797849, upper bound: 1.6173799
time: 7.09 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -9.7212629, -5.1725130, -9.7212772, -5.1725125, -3.7377477, 3.7364590
1: -17.1234303, -13.3353443, -17.1234398, -13.3353319, -3.4939613, 3.4955778
2: -8.1695328, -4.2664700, -8.1695595, -4.2664623, -3.3451805, 3.3651066
3: -13.8282261, -8.7362127, -13.8282642, -8.7362051, -5.0452681, 5.0469999
4: -3.9074385, -0.2440178, -3.9074695, -0.2440007, -3.2562590, 3.2806249
5: -13.9940996, -9.9756966, -13.9941082, -9.9756899, -2.9567499, 2.9484344
6: -15.9595490, -11.4040546, -15.9595528, -11.4040365, -3.8239841, 3.8400292
7: -8.3881712, -4.1793394, -8.3881950, -4.1793032, -4.0955029, 4.0859241
8: -6.7207527, -2.9882340, -6.7207656, -2.9882154, -3.7325373, 3.7325315
9: 3.9066682, 6.5659094, 3.9066486, 6.5659370, -2.6592689, 2.6592607

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 4557
type: A, layer: 1, pos: 5820
type: B, layer: 1, pos: 5820
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 906

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 961

## Relational analysis of IS_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5823409, upper bound: 1.6174521
time: 6.31 seconds

## Relational analysis of IS_A2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4557

## Relational analysis of IS_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5797849, upper bound: 1.6175682
time: 5.17 seconds

## Relational analysis of IS_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5797849, upper bound: 1.6175712
time: 6.01 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 44.76 seconds
IS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 44.76
Output dim: 9, lower bound: -1.5797849, upper bound: 1.6173785
IS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 44.76
Output dim: 9, lower bound: -1.5797849, upper bound: 1.6173799
IS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 44.76
Output dim: 9, lower bound: -1.5797849, upper bound: 1.6175682
IS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 44.76
Output dim: 9, lower bound: -1.5797849, upper bound: 1.6175712

## BFS IS instance: IS_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -9.7212629, -5.1725130, -9.6324787, -5.2194176, -3.5417542, 3.6523499
1: -17.1234303, -13.3353443, -17.0599537, -13.3951693, -3.4131441, 3.4354563
2: -8.1695328, -4.2664700, -8.0496616, -4.3364553, -3.1305265, 3.2464643
3: -13.8282261, -8.7362127, -13.7152596, -8.7844372, -4.9401093, 4.9790468
4: -3.9074385, -0.2440178, -3.8215551, -0.3094628, -3.2725830, 3.1954613
5: -13.9940996, -9.9756966, -13.9259109, -10.0381699, -2.8649178, 2.8880627
6: -15.9595490, -11.4040546, -15.8907461, -11.4718838, -3.6650114, 3.7551675
7: -8.3881712, -4.1793394, -8.2959061, -4.3132200, -3.9538469, 3.9035063
8: -6.7207527, -2.9882340, -6.6310477, -3.0642805, -3.6098971, 3.6428137
9: 3.9066682, 6.5659094, 4.0080853, 6.4742885, -2.5676203, 2.5578241

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 5820
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 5820
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 494

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 961

## Relational analysis of IS_A2_A2_B1_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5631269, upper bound: 1.6111431
time: 5.41 seconds

## Relational analysis of IS_A2_A2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5820

## Relational analysis of IS_A2_A2_B1_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5749133, upper bound: 1.6170496
time: 5.11 seconds

## Relational analysis of IS_A2_A2_B1_B1_B2

### Relational analysis result of IS_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5797791, upper bound: 1.6173730
time: 5.41 seconds

## BFS IS instance: IS_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -9.7212629, -5.1725130, -9.6517410, -5.2123356, -3.5664997, 3.6707110
1: -17.1234303, -13.3353443, -17.0863514, -13.3565893, -3.4468880, 3.4625945
2: -8.1695328, -4.2664700, -8.1016903, -4.3077345, -3.1557922, 3.2707911
3: -13.8282261, -8.7362127, -13.7873850, -8.7640591, -4.9597759, 5.0138540
4: -3.9074385, -0.2440178, -3.8931270, -0.2618673, -3.2450166, 3.2378945
5: -13.9940996, -9.9756966, -13.9539986, -9.9989147, -2.8920445, 2.9139519
6: -15.9595490, -11.4040546, -15.9145527, -11.4365282, -3.7295933, 3.8008585
7: -8.3881712, -4.1793394, -8.3611259, -4.2134304, -4.0450954, 4.0051689
8: -6.7207527, -2.9882340, -6.6736231, -3.0213170, -3.6424875, 3.6853890
9: 3.9066682, 6.5659094, 3.9322391, 6.5279784, -2.6213102, 2.6336703

Time for backsubstitution: 14.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5820
type: B, layer: 1, pos: 5820
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 494

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 961

## Relational analysis of IS_A2_A2_B1_B2_B1

### Relational analysis result of IS_A2_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5631269, upper bound: 1.6111446
time: 9.03 seconds

## Relational analysis of IS_A2_A2_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5820

## Relational analysis of IS_A2_A2_B1_B2_A1

### Relational analysis result of IS_A2_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5796569, upper bound: 1.6123691
time: 5.40 seconds

## Relational analysis of IS_A2_A2_B1_B2_A2

### Relational analysis result of IS_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5797792, upper bound: 1.6173736
time: 7.01 seconds

## BFS IS instance: IS_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -9.7212629, -5.1725130, -9.7020168, -5.1795983, -3.7129965, 3.7193818
1: -17.1234303, -13.3353443, -17.0970554, -13.3739710, -3.4601326, 3.4668269
2: -8.1695328, -4.2664700, -8.1175184, -4.2952156, -3.3222728, 3.3188810
3: -13.8282261, -8.7362127, -13.7561626, -8.7565775, -5.0255985, 5.0109501
4: -3.9074385, -0.2440178, -3.8358641, -0.2915483, -3.2875867, 3.2137623
5: -13.9940996, -9.9756966, -13.9660082, -10.0149679, -2.9212790, 2.9256499
6: -15.9595490, -11.4040546, -15.9357290, -11.4394979, -3.7752905, 3.7943172
7: -8.3881712, -4.1793394, -8.3226089, -4.2791243, -3.9946737, 3.9818144
8: -6.7207527, -2.9882340, -6.6781731, -3.0312710, -3.6894817, 3.6899390
9: 3.9066682, 6.5659094, 3.9826174, 6.5122128, -2.6055446, 2.5832920

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 5820
type: A, layer: 1, pos: 4597
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 494

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 961

## Relational analysis of IS_A2_A2_B2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5762638, upper bound: 1.6174519
time: 5.21 seconds

## Relational analysis of IS_A2_A2_B2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5765123, upper bound: 1.6175579
time: 4.98 seconds

## BFS IS instance: IS_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -9.7212629, -5.1725130, -9.7212629, -5.1725130, -3.7377424, 3.7377422
1: -17.1234303, -13.3353443, -17.1234303, -13.3353443, -3.4939489, 3.4939489
2: -8.1695328, -4.2664700, -8.1695328, -4.2664700, -3.3451738, 3.3451738
3: -13.8282261, -8.7362127, -13.8282261, -8.7362127, -5.0452614, 5.0452614
4: -3.9074385, -0.2440178, -3.9074385, -0.2440178, -3.2562408, 3.2562408
5: -13.9940996, -9.9756966, -13.9940996, -9.9756966, -2.9484262, 2.9484272
6: -15.9595490, -11.4040546, -15.9595490, -11.4040546, -3.8400030, 3.8400030
7: -8.3881712, -4.1793394, -8.3881712, -4.1793394, -4.0859022, 4.0859022
8: -6.7207527, -2.9882340, -6.7207527, -2.9882340, -3.7325187, 3.7325187
9: 3.9066682, 6.5659094, 3.9066682, 6.5659094, -2.6592412, 2.6592412

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 5820
type: B, layer: 1, pos: 5820
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 494

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 961

## Relational analysis of IS_A2_A2_B2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5762638, upper bound: 1.6174541
time: 7.22 seconds

## Relational analysis of IS_A2_A2_B2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5765128, upper bound: 1.6175616
time: 7.54 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 35.44 seconds
IS_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 35.44
Output dim: 9, lower bound: -1.5749133, upper bound: 1.6170496
IS_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 35.44
Output dim: 9, lower bound: -1.5797791, upper bound: 1.6173730
IS_A2_A2_B1_B2_A1, status: Status.VERIFIED, split count: 5, time: 35.44
Output dim: 9, lower bound: -1.5796569, upper bound: 1.6123691
IS_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 35.44
Output dim: 9, lower bound: -1.5797792, upper bound: 1.6173736
IS_A2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 35.44
Output dim: 9, lower bound: -1.5762638, upper bound: 1.6174519
IS_A2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 35.44
Output dim: 9, lower bound: -1.5765123, upper bound: 1.6175579
IS_A2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 35.44
Output dim: 9, lower bound: -1.5762638, upper bound: 1.6174541
IS_A2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 35.44
Output dim: 9, lower bound: -1.5765128, upper bound: 1.6175616

## BFS IS instance: IS_A2_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -9.7195625, -5.1758204, -9.6216021, -5.2360430, -3.5206060, 3.6303372
1: -17.1204777, -13.3362255, -17.0444679, -13.4006319, -3.4039989, 3.4181523
2: -8.1582336, -4.2667570, -7.9923553, -4.3435402, -3.0847292, 3.1885369
3: -13.8271523, -8.7430286, -13.7059441, -8.8195362, -4.9019051, 4.9584389
4: -3.9066076, -0.2501295, -3.8146389, -0.3409367, -3.2388268, 3.1801395
5: -13.9928942, -9.9781284, -13.9181099, -10.0508451, -2.8511176, 2.8711650
6: -15.9559441, -11.4052343, -15.8720627, -11.4799128, -3.6506472, 3.7341847
7: -8.3868999, -4.1918607, -8.2831631, -4.3762827, -3.8874922, 3.8491035
8: -6.7101378, -2.9892101, -6.5769100, -3.0744185, -3.5737839, 3.5876999
9: 3.9078569, 6.5609732, 4.0165462, 6.4490981, -2.5412412, 2.5444269

Time for backsubstitution: 14.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 4597
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 5820
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 494

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 961

## Relational analysis of IS_A2_A2_B1_B1_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5582586, upper bound: 1.6108133
time: 7.06 seconds

## Relational analysis of IS_A2_A2_B1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A2_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4597

## Relational analysis of IS_A2_A2_B1_B1_B1_A1

### Relational analysis result of IS_A2_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5740826, upper bound: 1.6170470
time: 5.24 seconds

## Relational analysis of IS_A2_A2_B1_B1_B1_A2

### Relational analysis result of IS_A2_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5749111, upper bound: 1.6170475
time: 5.33 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -9.7212629, -5.1725130, -9.6324778, -5.2194185, -3.5346413, 3.6513283
1: -17.1234303, -13.3353443, -17.0599556, -13.3951683, -3.4131398, 3.4300051
2: -8.1695328, -4.2664700, -8.0496588, -4.3364582, -3.1243362, 3.2120404
3: -13.8282261, -8.7362127, -13.7152576, -8.7844410, -4.9264536, 4.9782152
4: -3.9074385, -0.2440178, -3.8215582, -0.3094641, -3.2552404, 3.1954603
5: -13.9940996, -9.9756966, -13.9259090, -10.0381718, -2.8629694, 2.8870242
6: -15.9595490, -11.4040546, -15.8907433, -11.4718857, -3.6646624, 3.7455673
7: -8.3881712, -4.1793394, -8.2959051, -4.3132286, -3.9180527, 3.8985605
8: -6.7207527, -2.9882340, -6.6310444, -3.0642819, -3.6095104, 3.6428103
9: 3.9066682, 6.5659094, 4.0080872, 6.4742856, -2.5676174, 2.5578222

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 4597
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 5820
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 494

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 961

## Relational analysis of IS_A2_A2_B1_B1_B2_B1

### Relational analysis result of IS_A2_A2_B1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5631210, upper bound: 1.6111375
time: 5.46 seconds

## Relational analysis of IS_A2_A2_B1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A2_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4597

## Relational analysis of IS_A2_A2_B1_B1_B2_A1

### Relational analysis result of IS_A2_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5789516, upper bound: 1.6173709
time: 4.83 seconds

## Relational analysis of IS_A2_A2_B1_B1_B2_A2

### Relational analysis result of IS_A2_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5797769, upper bound: 1.6173706
time: 5.26 seconds

## BFS IS instance: IS_A2_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -9.7212629, -5.1725149, -9.6517410, -5.2123356, -3.5654926, 3.6635973
1: -17.1234283, -13.3353443, -17.0863514, -13.3565893, -3.4414372, 3.4625945
2: -8.1695271, -4.2664680, -8.1016903, -4.3077345, -3.1248226, 3.2687776
3: -13.8282299, -8.7362146, -13.7873850, -8.7640591, -4.9585094, 5.0000267
4: -3.9074388, -0.2440218, -3.8931270, -0.2618673, -3.2450166, 3.2205625
5: -13.9941006, -9.9756994, -13.9539986, -9.9989147, -2.8920445, 2.9120054
6: -15.9595451, -11.4040556, -15.9145527, -11.4365282, -3.7199078, 3.8005013
7: -8.3881693, -4.1793466, -8.3611259, -4.2134304, -4.0450964, 3.9693747
8: -6.7207460, -2.9882360, -6.6736231, -3.0213170, -3.6042471, 3.6853871
9: 3.9066687, 6.5659056, 3.9322391, 6.5279784, -2.6213098, 2.6336665

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 5820
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 494

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 961

## Relational analysis of IS_A2_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_A2_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5638789, upper bound: 1.6111382
time: 4.90 seconds

## Relational analysis of IS_A2_A2_B1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A2_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4597

## Relational analysis of IS_A2_A2_B1_B2_A2_A1

### Relational analysis result of IS_A2_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5796478, upper bound: 1.6173717
time: 5.16 seconds

## Relational analysis of IS_A2_A2_B1_B2_A2_A2

### Relational analysis result of IS_A2_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5805120, upper bound: 1.6173718
time: 4.72 seconds

## BFS IS instance: IS_A2_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -9.7149076, -5.1730995, -9.6778927, -5.1809702, -3.5087633, 3.6945286
1: -17.1228561, -13.3355999, -17.0951767, -13.3747149, -3.4032078, 3.4594250
2: -8.1673851, -4.2668152, -8.1093798, -4.2960010, -3.1259613, 3.3016803
3: -13.8247261, -8.7365637, -13.7428932, -8.7575684, -4.9191017, 4.9972982
4: -3.9072618, -0.2463772, -3.8353288, -0.3004847, -3.2775812, 3.1855431
5: -13.9905691, -9.9759216, -13.9526014, -10.0154686, -2.8399572, 2.9117279
6: -15.9553556, -11.4046059, -15.9198980, -11.4409752, -3.6411076, 3.7782040
7: -8.3878822, -4.1816225, -8.3218994, -4.2877769, -3.9855289, 3.9116836
8: -6.7161088, -2.9887400, -6.6605358, -3.0325308, -3.5800142, 3.6717958
9: 3.9069481, 6.5629158, 3.9832144, 6.5008535, -2.5939054, 2.5797014

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 4597
type: B, layer: 1, pos: 5820
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 494

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A2_A2_B2_B1_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5762638, upper bound: 1.5955435
time: 5.73 seconds

## Relational analysis of IS_A2_A2_B2_B1_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5762638, upper bound: 1.6174510
time: 4.89 seconds

## BFS IS instance: IS_A2_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -9.7212343, -5.1725154, -9.7024937, -5.1455774, -3.7135057, 3.7160363
1: -17.1234283, -13.3353729, -17.0980320, -13.3726158, -3.4677982, 3.4597273
2: -8.1695080, -4.2664680, -8.1178265, -4.2835922, -3.3208685, 3.3130214
3: -13.8282223, -8.7362165, -13.7570648, -8.7363491, -5.0344353, 5.0096989
4: -3.9074383, -0.2440379, -3.8501179, -0.2909718, -3.2858901, 3.2278514
5: -13.9940796, -9.9756994, -13.9662056, -9.9955311, -2.9266210, 2.9234564
6: -15.9595413, -11.4040689, -15.9361582, -11.4152050, -3.7973394, 3.7921319
7: -8.3881674, -4.1793680, -8.3382816, -4.2785730, -3.9937887, 3.9859281
8: -6.7207298, -2.9882374, -6.6786380, -3.0046229, -3.7161069, 3.6904006
9: 3.9067016, 6.5659037, 3.9652252, 6.5122957, -2.6055942, 2.6006784

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 5820
type: A, layer: 1, pos: 4597
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 494

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A2_A2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5820

## Relational analysis of IS_A2_A2_B2_B1_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5716558, upper bound: 1.6172241
time: 5.31 seconds

## Relational analysis of IS_A2_A2_B2_B1_B2_B2

### Relational analysis result of IS_A2_A2_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5765069, upper bound: 1.6175533
time: 5.41 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -9.7149076, -5.1730995, -9.6971388, -5.1738844, -3.5292521, 3.7128897
1: -17.1228561, -13.3355999, -17.1215343, -13.3360863, -3.4370122, 3.4865284
2: -8.1673851, -4.2668152, -8.1613903, -4.2672548, -3.1296864, 3.3301728
3: -13.8247261, -8.7365637, -13.8149700, -8.7371988, -4.9387665, 5.0316114
4: -3.9072618, -0.2463772, -3.9069009, -0.2529503, -3.2463393, 3.2279797
5: -13.9905691, -9.9759216, -13.9806919, -9.9762020, -2.8669667, 2.9347453
6: -15.9553556, -11.4046059, -15.9437113, -11.4055462, -3.7057343, 3.8238821
7: -8.3878822, -4.1816225, -8.3874245, -4.1879916, -4.0767593, 4.0016890
8: -6.7161088, -2.9887400, -6.7031193, -2.9894996, -3.6125612, 3.7143793
9: 3.9069481, 6.5629158, 3.9072676, 6.5545449, -2.6475968, 2.6556482

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5820
type: B, layer: 1, pos: 5820
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 4597
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 494

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A2_A2_B2_B2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5770454, upper bound: 1.5955459
time: 5.15 seconds

## Relational analysis of IS_A2_A2_B2_B2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5770454, upper bound: 1.6174542
time: 5.39 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -9.7212343, -5.1725154, -9.7217426, -5.1384907, -3.7382526, 3.7343969
1: -17.1234283, -13.3353729, -17.1244049, -13.3339958, -3.5016117, 3.4868474
2: -8.1695080, -4.2664680, -8.1698380, -4.2548652, -3.3459806, 3.3359780
3: -13.8282223, -8.7362165, -13.8291264, -8.7159901, -5.0654030, 5.0440102
4: -3.9074383, -0.2440379, -3.9216888, -0.2434660, -3.2545176, 3.2703300
5: -13.9940796, -9.9756994, -13.9942989, -9.9562759, -2.9563961, 2.9462066
6: -15.9595413, -11.4040689, -15.9600010, -11.3797379, -3.8603582, 3.8378162
7: -8.3881674, -4.1793680, -8.4038019, -4.1788039, -4.0850105, 4.1012545
8: -6.7207298, -2.9882374, -6.7212281, -2.9615769, -3.7591529, 3.7329907
9: 3.9067016, 6.5659037, 3.8893032, 6.5659900, -2.6592884, 2.6766005

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 5820
type: A, layer: 1, pos: 5820
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 494

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A2_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5820

## Relational analysis of IS_A2_A2_B2_B2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5723460, upper bound: 1.6172249
time: 5.25 seconds

## Relational analysis of IS_A2_A2_B2_B2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5773416, upper bound: 1.6175567
time: 6.18 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 32.17 seconds
IS_A2_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 32.17
Output dim: 9, lower bound: -1.5740826, upper bound: 1.6170470
IS_A2_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 32.17
Output dim: 9, lower bound: -1.5749111, upper bound: 1.6170475
IS_A2_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 32.17
Output dim: 9, lower bound: -1.5789516, upper bound: 1.6173709
IS_A2_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 32.17
Output dim: 9, lower bound: -1.5797769, upper bound: 1.6173706
IS_A2_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 32.17
Output dim: 9, lower bound: -1.5796478, upper bound: 1.6173717
IS_A2_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 32.17
Output dim: 9, lower bound: -1.5805120, upper bound: 1.6173718
IS_A2_A2_B2_B1_B1_A1, status: Status.VERIFIED, split count: 6, time: 32.17
Output dim: 9, lower bound: -1.5762638, upper bound: 1.5955435
IS_A2_A2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 32.17
Output dim: 9, lower bound: -1.5762638, upper bound: 1.6174510
IS_A2_A2_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 32.17
Output dim: 9, lower bound: -1.5716558, upper bound: 1.6172241
IS_A2_A2_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 32.17
Output dim: 9, lower bound: -1.5765069, upper bound: 1.6175533
IS_A2_A2_B2_B2_B1_A1, status: Status.VERIFIED, split count: 6, time: 32.17
Output dim: 9, lower bound: -1.5770454, upper bound: 1.5955459
IS_A2_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 32.17
Output dim: 9, lower bound: -1.5770454, upper bound: 1.6174542
IS_A2_A2_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 32.17
Output dim: 9, lower bound: -1.5723460, upper bound: 1.6172249
IS_A2_A2_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 32.17
Output dim: 9, lower bound: -1.5773416, upper bound: 1.6175567

## BFS IS instance: IS_A2_A2_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -9.7083931, -5.2109857, -9.6206980, -5.2393570, -3.4874134, 3.5944710
1: -17.1056595, -13.3489857, -17.0431919, -13.4016743, -3.3739600, 3.3987889
2: -8.0534077, -4.2851110, -7.9826384, -4.3448205, -2.9782934, 3.1017056
3: -13.7067585, -8.7722759, -13.6948090, -8.8216171, -4.7784843, 4.8680563
4: -3.8883996, -0.2978833, -3.8132823, -0.3454809, -3.1878319, 3.1284761
5: -13.9776030, -10.0375757, -13.9170170, -10.0563459, -2.7874594, 2.8009484
6: -15.9020872, -11.4226589, -15.8671036, -11.4811974, -3.5952682, 3.6913505
7: -8.3519077, -4.2105293, -8.2798805, -4.3777919, -3.8353672, 3.8000622
8: -6.6946783, -3.0017385, -6.5752950, -3.0755949, -3.5578918, 3.5735564
9: 3.9181242, 6.5555148, 4.0174837, 6.4486647, -2.5305405, 2.5380311

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 4597

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 961

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_B1

### Relational analysis result of IS_A2_A2_B1_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5574304, upper bound: 1.6108111
time: 5.17 seconds

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 494

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_B1

### Relational analysis result of IS_A2_A2_B1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5738838, upper bound: 1.6170367
time: 5.18 seconds

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_B2

### Relational analysis result of IS_A2_A2_B1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5740725, upper bound: 1.6170365
time: 5.33 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -9.7195606, -5.1758237, -9.6216021, -5.2360430, -3.5194139, 3.6202085
1: -17.1204796, -13.3362217, -17.0444679, -13.4006319, -3.4039950, 3.4388809
2: -8.1582327, -4.2667580, -7.9923553, -4.3435402, -3.0677252, 3.1858132
3: -13.8271427, -8.7430286, -13.7059441, -8.8195362, -4.9004793, 4.9584370
4: -3.9066057, -0.2501335, -3.8146389, -0.3409367, -3.2378139, 3.1663766
5: -13.9928932, -9.9781322, -13.9181099, -10.0508451, -2.8496413, 2.8554058
6: -15.9559412, -11.4052334, -15.8720627, -11.4799128, -3.6506424, 3.7464981
7: -8.3868980, -4.1918612, -8.2831631, -4.3762827, -3.8927078, 3.8481622
8: -6.7101355, -2.9892120, -6.5769100, -3.0744185, -3.5828991, 3.5876980
9: 3.9078565, 6.5609732, 4.0165462, 6.4490981, -2.5412416, 2.5444269

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 4597

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 961

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_B1

### Relational analysis result of IS_A2_A2_B1_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5582563, upper bound: 1.6108115
time: 4.97 seconds

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 494

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_B1

### Relational analysis result of IS_A2_A2_B1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5747123, upper bound: 1.6170369
time: 6.17 seconds

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_B2

### Relational analysis result of IS_A2_A2_B1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5749013, upper bound: 1.6170369
time: 5.38 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -9.7101173, -5.2076912, -9.6315813, -5.2227407, -3.5013833, 3.6154513
1: -17.1086082, -13.3481112, -17.0586758, -13.3962088, -3.3830938, 3.4106340
2: -8.0646982, -4.2848182, -8.0399370, -4.3377399, -3.0178642, 3.1300812
3: -13.7078304, -8.7654629, -13.7041264, -8.7865248, -4.8031082, 4.8987589
4: -3.8892407, -0.2917700, -3.8202050, -0.3140072, -3.2042823, 3.1428952
5: -13.9788322, -10.0351410, -13.9248857, -10.0436678, -2.7993574, 2.8168180
6: -15.9056950, -11.4214687, -15.8857832, -11.4731636, -3.6092949, 3.7049913
7: -8.3531666, -4.1980009, -8.2926207, -4.3147354, -3.8659163, 3.8494649
8: -6.7052956, -3.0007629, -6.6294346, -3.0654554, -3.5935907, 3.6286716
9: 3.9169369, 6.5604491, 4.0090246, 6.4738536, -2.5569167, 2.5514245

Time for backsubstitution: 14.58 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=6, k_high=6, k_mid=6, eps_mid=0.0234375, abs_max=2.6592936515808105
rel_dist={9: [-1.6176245637311917, 1.6176249981850015]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01953125
execution time: 2417.66 seconds
