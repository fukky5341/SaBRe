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
execution time: IAR + LP analysis = 15.19 + 33.64 = 48.83 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -2.0738973, upper bound: 2.0738940


# Binary Search by BASE starts (time budget: 3551.17 seconds, max iter: 100)

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
Binary search time: 208.08 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01953125


# Individual Split (IS_dual_ind) starts
Time budget: 3343.08 seconds

## Binary search (step 0) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5875
type: A, layer: 1, pos: 4557
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5875

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8581708, upper bound: 1.8315596
time: 4.96 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8604479, upper bound: 1.8604462
time: 4.86 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 9.99 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 9.99
Output dim: 9, lower bound: -1.8581708, upper bound: 1.8315596
IS_A2, status: Status.UNKNOWN, split count: 1, time: 9.99
Output dim: 9, lower bound: -1.8604479, upper bound: 1.8604462

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -9.6517572, -5.2123337, -9.7076283, -5.1739035, -3.9657001, 3.9833193
1: -17.0863571, -13.3565741, -17.1163940, -13.3358536, -3.7505035, 3.7598200
2: -8.1017151, -4.3077283, -8.1562595, -4.2671590, -3.4665747, 3.4805641
3: -13.7874165, -8.7640533, -13.8203630, -8.7370911, -5.0503254, 5.0563097
4: -3.8931594, -0.2618493, -3.9069490, -0.2473956, -3.6062307, 3.6069317
5: -13.9540033, -9.9989052, -13.9862242, -9.9761734, -3.3139582, 3.3240514
6: -15.9145575, -11.4365091, -15.9506664, -11.4054737, -4.1027718, 4.1076741
7: -8.3611498, -4.2133913, -8.3875217, -4.1859102, -4.1752396, 4.1741304
8: -6.6736360, -3.0213003, -6.7115211, -2.9894505, -3.6841855, 3.6902208
9: 3.9322205, 6.5280085, 3.9072409, 6.5584183, -2.6261978, 2.6207676

Time for backsubstitution: 14.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4557
type: B, layer: 1, pos: 5875
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 5820
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 494

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4557

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8318899, upper bound: 1.8314931
time: 4.51 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8581027, upper bound: 1.8314929
time: 4.53 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -9.7212772, -5.1725125, -9.7212811, -5.1725097, -4.1626396, 4.1678424
1: -17.1234398, -13.3353319, -17.1234474, -13.3353291, -3.7881107, 3.7881155
2: -8.1695595, -4.2664623, -8.1695728, -4.2664623, -3.6541157, 3.6593037
3: -13.8282642, -8.7362051, -13.8282661, -8.7362089, -5.0920553, 5.0920610
4: -3.9074695, -0.2440007, -3.9074690, -0.2440012, -3.6350198, 3.6288652
5: -13.9941082, -9.9756899, -13.9941177, -9.9756889, -3.3817692, 3.3869648
6: -15.9595528, -11.4040365, -15.9595585, -11.4040346, -4.2238474, 4.2286806
7: -8.3881950, -4.1793032, -8.3881950, -4.1792917, -4.2089033, 4.2088919
8: -6.7207656, -2.9882154, -6.7207699, -2.9882140, -3.7325516, 3.7325544
9: 3.9066486, 6.5659370, 3.9066472, 6.5659409, -2.6592922, 2.6592898

Time for backsubstitution: 15.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5875
type: B, layer: 1, pos: 4557
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 5820
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 494

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5875

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8315608, upper bound: 1.8581702
time: 4.84 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8315602, upper bound: 1.8604475
time: 4.74 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 24.88 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 24.88
Output dim: 9, lower bound: -1.8318899, upper bound: 1.8314931
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 24.88
Output dim: 9, lower bound: -1.8581027, upper bound: 1.8314929
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 24.88
Output dim: 9, lower bound: -1.8315608, upper bound: 1.8581702
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 24.88
Output dim: 9, lower bound: -1.8315602, upper bound: 1.8604475

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -9.6481819, -5.2132645, -9.6883631, -5.1809926, -3.9440403, 3.9652100
1: -17.0814133, -13.3621302, -17.0900078, -13.3744850, -3.7069283, 3.7278776
2: -8.0908909, -4.3107691, -8.1042118, -4.2959070, -3.4262133, 3.4279423
3: -13.7726631, -8.7662458, -13.7482491, -8.7574615, -5.0152016, 4.9820032
4: -3.8792372, -0.2679958, -3.8353460, -0.2949450, -3.5842922, 3.5373230
5: -13.9507809, -10.0062990, -13.9581175, -10.0154495, -3.2766771, 3.2920365
6: -15.9115047, -11.4439316, -15.9268389, -11.4409285, -4.0465355, 4.0555353
7: -8.3503389, -4.2319884, -8.3219719, -4.2857313, -4.0646076, 4.0899835
8: -6.6683893, -3.0296516, -6.6689262, -3.0324993, -3.6358900, 3.6392746
9: 3.9409103, 6.5174379, 3.9831934, 6.5046940, -2.5637836, 2.5342445

Time for backsubstitution: 14.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4557
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4557

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8318899, upper bound: 1.8053087
time: 4.85 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8318899, upper bound: 1.8314935
time: 4.31 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -9.6517572, -5.2123337, -9.7076111, -5.1739049, -3.9644508, 3.9839783
1: -17.0863571, -13.3565741, -17.1163864, -13.3358660, -3.7504911, 3.7598124
2: -8.1017151, -4.3077283, -8.1562300, -4.2671652, -3.4665689, 3.4586616
3: -13.7874165, -8.7640533, -13.8203268, -8.7370939, -5.0503225, 5.0562735
4: -3.8931594, -0.2618493, -3.9069185, -0.2474124, -3.6062136, 3.5865993
5: -13.9540033, -9.9989052, -13.9862080, -9.9761801, -3.3070240, 3.3240452
6: -15.9145575, -11.4365091, -15.9506598, -11.4054956, -4.1157103, 4.1002188
7: -8.3611498, -4.2133913, -8.3874989, -4.1859493, -4.1752005, 4.1741076
8: -6.6736360, -3.0213003, -6.7115073, -2.9894705, -3.6841655, 3.6902070
9: 3.9322205, 6.5280085, 3.9072618, 6.5583878, -2.6261673, 2.6207466

Time for backsubstitution: 14.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4557
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4557

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8581034, upper bound: 1.8053105
time: 4.59 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8581034, upper bound: 1.8314926
time: 4.55 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -9.7212772, -5.1725125, -9.6517572, -5.2123337, -3.9971676, 4.0977230
1: -17.1234398, -13.3353319, -17.0863571, -13.3565741, -3.7668657, 3.7510252
2: -8.1695595, -4.2664623, -8.1017151, -4.3077283, -3.4938860, 3.5842667
3: -13.8282642, -8.7362051, -13.7874165, -8.7640533, -5.0642109, 5.0512114
4: -3.9074695, -0.2440007, -3.8931594, -0.2618493, -3.6164103, 3.6107116
5: -13.9941082, -9.9756899, -13.9540033, -9.9989052, -3.3320537, 3.3462548
6: -15.9595528, -11.4040365, -15.9145575, -11.4365091, -4.1166010, 4.1837182
7: -8.3881950, -4.1793032, -8.3611498, -4.2133913, -4.1748037, 4.1818466
8: -6.7207656, -2.9882154, -6.6736360, -3.0213003, -3.6994653, 3.6854205
9: 3.9066486, 6.5659370, 3.9322205, 6.5280085, -2.6213598, 2.6337166

Time for backsubstitution: 14.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4557
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 494

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4557

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8314930, upper bound: 1.8318912
time: 4.66 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8314935, upper bound: 1.8581018
time: 4.60 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -9.7212772, -5.1725125, -9.7212772, -5.1725125, -4.1626396, 4.1626401
1: -17.1234398, -13.3353319, -17.1234398, -13.3353319, -3.7881079, 3.7881079
2: -8.1695595, -4.2664623, -8.1695595, -4.2664623, -3.6592960, 3.6592965
3: -13.8282642, -8.7362051, -13.8282642, -8.7362051, -5.0920591, 5.0920591
4: -3.9074695, -0.2440007, -3.9074695, -0.2440007, -3.6288652, 3.6288648
5: -13.9941082, -9.9756899, -13.9941082, -9.9756899, -3.3817692, 3.3817692
6: -15.9595528, -11.4040365, -15.9595528, -11.4040365, -4.2238464, 4.2238464
7: -8.3881950, -4.1793032, -8.3881950, -4.1793032, -4.2088919, 4.2088919
8: -6.7207656, -2.9882154, -6.7207656, -2.9882154, -3.7325501, 3.7325501
9: 3.9066486, 6.5659370, 3.9066486, 6.5659370, -2.6592884, 2.6592884

Time for backsubstitution: 15.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4557
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4557

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8314930, upper bound: 1.8341444
time: 4.50 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8314930, upper bound: 1.8603793
time: 4.57 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 24.26 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 24.26
Output dim: 9, lower bound: -1.8318899, upper bound: 1.8053087
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 24.26
Output dim: 9, lower bound: -1.8318899, upper bound: 1.8314935
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 24.26
Output dim: 9, lower bound: -1.8581034, upper bound: 1.8053105
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 24.26
Output dim: 9, lower bound: -1.8581034, upper bound: 1.8314926
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 24.26
Output dim: 9, lower bound: -1.8314930, upper bound: 1.8318912
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 24.26
Output dim: 9, lower bound: -1.8314935, upper bound: 1.8581018
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 24.26
Output dim: 9, lower bound: -1.8314930, upper bound: 1.8341444
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 24.26
Output dim: 9, lower bound: -1.8314930, upper bound: 1.8603793

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -9.6324787, -5.2194176, -9.6883631, -5.1809926, -3.9275951, 3.9453316
1: -17.0599537, -13.3951693, -17.0900078, -13.3744850, -3.6854687, 3.6948385
2: -8.0496616, -4.3364553, -8.1042118, -4.2959070, -3.3874750, 3.4014926
3: -13.7152596, -8.7844372, -13.7482491, -8.7574615, -4.9577980, 4.9638119
4: -3.8215551, -0.3094628, -3.8353460, -0.2949450, -3.5266101, 3.5258832
5: -13.9259109, -10.0381699, -13.9581175, -10.0154495, -3.2528906, 3.2629862
6: -15.8907461, -11.4718838, -15.9268389, -11.4409285, -4.0179329, 4.0228839
7: -8.2959061, -4.3132200, -8.3219719, -4.2857313, -4.0101748, 4.0087519
8: -6.6310477, -3.0642805, -6.6689262, -3.0324993, -3.5985484, 3.6046457
9: 4.0080853, 6.4742885, 3.9831934, 6.5046940, -2.4966087, 2.4910951

Time for backsubstitution: 14.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5875
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 5820
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5875

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053113, upper bound: 1.8053122
time: 4.65 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053112, upper bound: 1.8053129
time: 5.02 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -9.6516857, -5.2123361, -9.6883631, -5.1809926, -3.9487247, 3.9647126
1: -17.0863495, -13.3565950, -17.0900078, -13.3744850, -3.7118645, 3.7334127
2: -8.1016855, -4.3077364, -8.1042118, -4.2959070, -3.4366074, 3.4311352
3: -13.7873831, -8.7640629, -13.7482491, -8.7574615, -5.0299215, 4.9841862
4: -3.8931284, -0.2618936, -3.8353460, -0.2949450, -3.5981834, 3.5433345
5: -13.9539986, -9.9989195, -13.9581175, -10.0154495, -3.2792215, 3.2995620
6: -15.9145517, -11.4365759, -15.9268389, -11.4409285, -4.0475903, 4.0620131
7: -8.3611193, -4.2134285, -8.3219719, -4.2857313, -4.0753880, 4.1085434
8: -6.6736207, -3.0213208, -6.6689262, -3.0324993, -3.6411214, 3.6476054
9: 3.9322443, 6.5279779, 3.9831934, 6.5046940, -2.5724497, 2.5447845

Time for backsubstitution: 15.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5875
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 5820
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 494

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5875

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053113, upper bound: 1.8314925
time: 4.95 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053113, upper bound: 1.8314939
time: 5.07 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -9.6324787, -5.2194176, -9.7076111, -5.1739049, -3.9470024, 3.9666777
1: -17.0599537, -13.3951693, -17.1163864, -13.3358660, -3.7240877, 3.7212172
2: -8.0496616, -4.3364553, -8.1562300, -4.2671652, -3.4169798, 3.4506207
3: -13.7152596, -8.7844372, -13.8203268, -8.7370939, -4.9781656, 5.0358896
4: -3.8215551, -0.3094628, -3.9069185, -0.2474124, -3.5393972, 3.5974557
5: -13.9259109, -10.0381699, -13.9862080, -9.9761801, -3.2876115, 3.2893376
6: -15.8907461, -11.4718838, -15.9506598, -11.4054956, -4.0656900, 4.0525637
7: -8.2959061, -4.3132200, -8.3874989, -4.1859493, -4.1099567, 4.0742788
8: -6.6310477, -3.0642805, -6.7115073, -2.9894705, -3.6415772, 3.6472268
9: 4.0080853, 6.4742885, 3.9072618, 6.5583878, -2.5503025, 2.5670266

Time for backsubstitution: 15.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5875
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 5820
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 494

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5875

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053113, upper bound: 1.8053082
time: 4.93 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053112, upper bound: 1.8053080
time: 5.13 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -9.6517410, -5.2123356, -9.7076111, -5.1739049, -3.9663219, 3.9839721
1: -17.0863514, -13.3565893, -17.1163864, -13.3358660, -3.7504854, 3.7597971
2: -8.1016903, -4.3077345, -8.1562300, -4.2671652, -3.4446630, 3.4586554
3: -13.7873850, -8.7640591, -13.8203268, -8.7370939, -5.0502911, 5.0562677
4: -3.8931270, -0.2618673, -3.9069185, -0.2474124, -3.5858822, 3.5865808
5: -13.9539986, -9.9989147, -13.9862080, -9.9761801, -3.3070183, 3.3171096
6: -15.9145527, -11.4365282, -15.9506598, -11.4054956, -4.1156883, 4.1205921
7: -8.3611259, -4.2134304, -8.3874989, -4.1859493, -4.1751766, 4.1740685
8: -6.6736231, -3.0213170, -6.7115073, -2.9894705, -3.6841526, 3.6901903
9: 3.9322391, 6.5279784, 3.9072618, 6.5583878, -2.6261487, 2.6207166

Time for backsubstitution: 14.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5875
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 5820
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 494

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5875

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053112, upper bound: 1.8314951
time: 4.64 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053113, upper bound: 1.8314937
time: 5.17 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -9.7020168, -5.1795983, -9.6481819, -5.2132645, -3.9790850, 4.0757289
1: -17.0970554, -13.3739710, -17.0814133, -13.3621302, -3.7349253, 3.7074423
2: -8.1175184, -4.2952156, -8.0908909, -4.3107691, -3.4412594, 3.5440431
3: -13.7561626, -8.7565775, -13.7726631, -8.7662458, -4.9899168, 5.0160856
4: -3.8358641, -0.2915483, -3.8792372, -0.2679958, -3.5468040, 3.5876889
5: -13.9660082, -10.0149679, -13.9507809, -10.0062990, -3.3000369, 3.3088608
6: -15.9357290, -11.4394979, -15.9115047, -11.4439316, -4.0644627, 4.1274014
7: -8.3226089, -4.2791243, -8.3503389, -4.2319884, -4.0906205, 4.0712147
8: -6.6781731, -3.0312710, -6.6683893, -3.0296516, -3.6485214, 3.6371183
9: 3.9826174, 6.5122128, 3.9409103, 6.5174379, -2.5348206, 2.5713024

Time for backsubstitution: 15.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4557
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 5820
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 494

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4557

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053082, upper bound: 1.8318888
time: 4.77 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053082, upper bound: 1.8318889
time: 4.87 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -9.7212629, -5.1725130, -9.6517572, -5.2123337, -3.9978333, 4.0964718
1: -17.1234303, -13.3353443, -17.0863571, -13.3565741, -3.7668562, 3.7510128
2: -8.1695328, -4.2664700, -8.1017151, -4.3077283, -3.4719830, 3.5842609
3: -13.8282261, -8.7362127, -13.7874165, -8.7640533, -5.0641727, 5.0512037
4: -3.9074385, -0.2440178, -3.8931594, -0.2618493, -3.5960789, 3.6106944
5: -13.9940996, -9.9756966, -13.9540033, -9.9989052, -3.3320479, 3.3393192
6: -15.9595490, -11.4040546, -15.9145575, -11.4365091, -4.1091442, 4.1966648
7: -8.3881712, -4.1793394, -8.3611498, -4.2133913, -4.1747799, 4.1818104
8: -6.7207527, -2.9882340, -6.6736360, -3.0213003, -3.6994524, 3.6854019
9: 3.9066682, 6.5659094, 3.9322205, 6.5280085, -2.6213403, 2.6336889

Time for backsubstitution: 14.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4557
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 5820
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4557

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053082, upper bound: 1.8581017
time: 4.68 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053083, upper bound: 1.8581024
time: 5.49 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -9.7020168, -5.1795983, -9.7177067, -5.1734443, -4.1445532, 4.1410851
1: -17.0970554, -13.3739710, -17.1184940, -13.3408947, -3.7561607, 3.7445230
2: -8.1175184, -4.2952156, -8.1587372, -4.2695093, -3.6066027, 3.6188726
3: -13.7561626, -8.7565775, -13.8135185, -8.7384005, -5.0177622, 5.0569410
4: -3.8358641, -0.2915483, -3.8935461, -0.2501363, -3.5592604, 3.6019979
5: -13.9660082, -10.0149679, -13.9908819, -9.9830856, -3.3496900, 3.3443723
6: -15.9357290, -11.4394979, -15.9565001, -11.4114771, -4.1716413, 4.1675310
7: -8.3226089, -4.2791243, -8.3773527, -4.1978941, -4.1247149, 4.0982285
8: -6.6781731, -3.0312710, -6.7155199, -2.9965792, -3.6815939, 3.6842489
9: 3.9826174, 6.5122128, 3.9153605, 6.5553665, -2.5727491, 2.5968523

Time for backsubstitution: 14.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4557
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 5820
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4557

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8065950, upper bound: 1.8341435
time: 4.50 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8065950, upper bound: 1.8341431
time: 4.85 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.7212629, -5.1725130, -9.7212772, -5.1725125, -4.1633024, 4.1613894
1: -17.1234303, -13.3353443, -17.1234398, -13.3353319, -3.7880983, 3.7880955
2: -8.1695328, -4.2664700, -8.1695595, -4.2664623, -3.6373920, 3.6592889
3: -13.8282261, -8.7362127, -13.8282642, -8.7362051, -5.0920210, 5.0920515
4: -3.9074385, -0.2440178, -3.9074695, -0.2440007, -3.6085348, 3.6288481
5: -13.9940996, -9.9756966, -13.9941082, -9.9756899, -3.3817625, 3.3748326
6: -15.9595490, -11.4040546, -15.9595528, -11.4040365, -4.2163887, 4.2367797
7: -8.3881712, -4.1793394, -8.3881950, -4.1793032, -4.2088680, 4.2088556
8: -6.7207527, -2.9882340, -6.7207656, -2.9882154, -3.7325373, 3.7325315
9: 3.9066682, 6.5659094, 3.9066486, 6.5659370, -2.6592689, 2.6592607

Time for backsubstitution: 14.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4557
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 5820
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4557

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8065950, upper bound: 1.8603826
time: 4.42 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8065950, upper bound: 1.8603806
time: 7.92 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 27.47 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 27.47
Output dim: 9, lower bound: -1.8053113, upper bound: 1.8053122
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 27.47
Output dim: 9, lower bound: -1.8053112, upper bound: 1.8053129
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 27.47
Output dim: 9, lower bound: -1.8053113, upper bound: 1.8314925
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.47
Output dim: 9, lower bound: -1.8053113, upper bound: 1.8314939
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 27.47
Output dim: 9, lower bound: -1.8053113, upper bound: 1.8053082
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 27.47
Output dim: 9, lower bound: -1.8053112, upper bound: 1.8053080
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 27.47
Output dim: 9, lower bound: -1.8053112, upper bound: 1.8314951
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.47
Output dim: 9, lower bound: -1.8053113, upper bound: 1.8314937
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 27.47
Output dim: 9, lower bound: -1.8053082, upper bound: 1.8318888
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 27.47
Output dim: 9, lower bound: -1.8053082, upper bound: 1.8318889
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 27.47
Output dim: 9, lower bound: -1.8053082, upper bound: 1.8581017
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.47
Output dim: 9, lower bound: -1.8053083, upper bound: 1.8581024
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 27.47
Output dim: 9, lower bound: -1.8065950, upper bound: 1.8341435
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 27.47
Output dim: 9, lower bound: -1.8065950, upper bound: 1.8341431
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 27.47
Output dim: 9, lower bound: -1.8065950, upper bound: 1.8603826
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.47
Output dim: 9, lower bound: -1.8065950, upper bound: 1.8603806

## BFS IS instance: IS_A1_B1_A1_B1

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

Time for backsubstitution: 15.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5798

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8020354, upper bound: 1.8053095
time: 4.70 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053124, upper bound: 1.8053092
time: 4.73 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -9.6324787, -5.2194176, -9.7019901, -5.1808753, -3.9277201, 3.9590964
1: -17.0599537, -13.3951693, -17.0970135, -13.3743057, -3.6856480, 3.7018442
2: -8.0496616, -4.3364553, -8.1175013, -4.2958589, -3.3838582, 3.4147921
3: -13.7152596, -8.7844372, -13.7561131, -8.7570534, -4.9582062, 4.9716759
4: -3.8215551, -0.3094628, -3.8356650, -0.2915885, -3.5299666, 3.5262022
5: -13.9259109, -10.0381699, -13.9660034, -10.0154209, -3.2523303, 3.2709799
6: -15.8907461, -11.4718838, -15.9357185, -11.4404163, -4.0185184, 4.0317755
7: -8.2959061, -4.3132200, -8.3221769, -4.2791514, -4.0167546, 4.0089569
8: -6.6310477, -3.0642805, -6.6781597, -3.0321503, -3.5988975, 3.6138792
9: 4.0080853, 6.4742885, 3.9829741, 6.5122108, -2.5041256, 2.4913144

Time for backsubstitution: 15.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5798

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8020353, upper bound: 1.8053095
time: 4.86 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053123, upper bound: 1.8053100
time: 5.14 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9.6516857, -5.2123361, -9.6324787, -5.2194176, -3.9103117, 3.9085884
1: -17.0863495, -13.3565950, -17.0599537, -13.3951693, -3.6911802, 3.7033587
2: -8.1016855, -4.3077364, -8.0496616, -4.3364553, -3.3960724, 3.3765850
3: -13.7873831, -8.7640629, -13.7152596, -8.7844372, -5.0029459, 4.9511967
4: -3.8931284, -0.2618936, -3.8215551, -0.3094628, -3.5836656, 3.5288191
5: -13.9539986, -9.9989195, -13.9259109, -10.0381699, -3.2567749, 3.2670197
6: -15.9145517, -11.4365759, -15.8907461, -11.4718838, -4.0165377, 4.0260091
7: -8.3611193, -4.2134285, -8.2959061, -4.3132200, -4.0478992, 4.0824776
8: -6.6736207, -3.0213208, -6.6310477, -3.0642805, -3.6093402, 3.6097269
9: 3.9322443, 6.5279779, 4.0080853, 6.4742885, -2.5420442, 2.5198927

Time for backsubstitution: 15.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 494

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5798

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8020313, upper bound: 1.8314888
time: 4.43 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053083, upper bound: 1.8314885
time: 4.80 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.6516857, -5.2123361, -9.7019901, -5.1808753, -3.9473667, 3.9784718
1: -17.0863495, -13.3565950, -17.0970135, -13.3743057, -3.7120438, 3.7404184
2: -8.1016855, -4.3077364, -8.1175013, -4.2958589, -3.4166136, 3.4444342
3: -13.7873831, -8.7640629, -13.7561131, -8.7570534, -5.0303297, 4.9920502
4: -3.8931284, -0.2618936, -3.8356650, -0.2915885, -3.6015399, 3.5429196
5: -13.9539986, -9.9989195, -13.9660034, -10.0154209, -3.2786622, 3.3075552
6: -15.9145517, -11.4365759, -15.9357185, -11.4404163, -4.0481749, 4.0709052
7: -8.3611193, -4.2134285, -8.3221769, -4.2791514, -4.0819678, 4.1087484
8: -6.6736207, -3.0213208, -6.6781597, -3.0321503, -3.6414704, 3.6568389
9: 3.9322443, 6.5279779, 3.9829741, 6.5122108, -2.5799665, 2.5450039

Time for backsubstitution: 15.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 494

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5798

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8020312, upper bound: 1.8314887
time: 4.69 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053083, upper bound: 1.8314894
time: 5.30 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.6324787, -5.2194176, -9.6517410, -5.2123356, -3.9085884, 3.9105687
1: -17.0599537, -13.3951693, -17.0863514, -13.3565893, -3.7033644, 3.6911821
2: -8.0496616, -4.3364553, -8.1016903, -4.3077345, -3.3764277, 3.3960767
3: -13.7152596, -8.7844372, -13.7873850, -8.7640591, -4.9512005, 5.0029478
4: -3.8215551, -0.3094628, -3.8931270, -0.2618673, -3.5255713, 3.5836642
5: -13.9259109, -10.0381699, -13.9539986, -9.9989147, -3.2651429, 3.2567754
6: -15.8907461, -11.4718838, -15.9145527, -11.4365282, -4.0345745, 4.0165386
7: -8.2959061, -4.3132200, -8.3611259, -4.2134304, -4.0824757, 4.0479059
8: -6.6310477, -3.0642805, -6.6736231, -3.0213170, -3.6097307, 3.6093426
9: 4.0080853, 6.4742885, 3.9322391, 6.5279784, -2.5198932, 2.5420494

Time for backsubstitution: 14.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5798

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8282759, upper bound: 1.8053047
time: 4.99 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8314922, upper bound: 1.8053055
time: 5.09 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.6324787, -5.2194176, -9.7212362, -5.1737866, -3.9471283, 3.9804282
1: -17.0599537, -13.3951693, -17.1233864, -13.3356800, -3.7242737, 3.7282171
2: -8.0496616, -4.3364553, -8.1695175, -4.2671132, -3.4048185, 3.4585326
3: -13.7152596, -8.7844372, -13.8281803, -8.7366867, -4.9785728, 5.0437431
4: -3.8215551, -0.3094628, -3.9072380, -0.2440574, -3.5433359, 3.5977752
5: -13.9259109, -10.0381699, -13.9940891, -9.9761477, -3.2870722, 3.2973347
6: -15.8907461, -11.4718838, -15.9595337, -11.4049778, -4.0662823, 4.0614543
7: -8.2959061, -4.3132200, -8.3877459, -4.1793680, -4.1165380, 4.0745258
8: -6.6310477, -3.0642805, -6.7207437, -2.9891171, -3.6419306, 3.6564631
9: 4.0080853, 6.4742885, 3.9070263, 6.5659070, -2.5578218, 2.5672622

Time for backsubstitution: 15.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 494

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5798

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8282754, upper bound: 1.8053049
time: 4.84 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8314916, upper bound: 1.8053049
time: 5.10 seconds

## BFS IS instance: IS_A1_B2_A2_B1

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

Time for backsubstitution: 14.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 494

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5798

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8020334, upper bound: 1.8314905
time: 4.90 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053082, upper bound: 1.8314898
time: 4.96 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -9.6517410, -5.2123356, -9.7212362, -5.1737866, -3.9664459, 3.9977102
1: -17.0863514, -13.3565893, -17.1233864, -13.3356800, -3.7506714, 3.7667971
2: -8.1016903, -4.3077345, -8.1695175, -4.2671132, -3.4409137, 3.4719601
3: -13.7873850, -8.7640591, -13.8281803, -8.7366867, -5.0506983, 5.0641212
4: -3.8931270, -0.2618673, -3.9072380, -0.2440574, -3.5898228, 3.5861697
5: -13.9539986, -9.9989147, -13.9940891, -9.9761477, -3.3064780, 3.3251061
6: -15.9145527, -11.4365282, -15.9595337, -11.4049778, -4.1162872, 4.1294856
7: -8.3611259, -4.2134304, -8.3877459, -4.1793680, -4.1817579, 4.1743155
8: -6.6736231, -3.0213170, -6.7207437, -2.9891171, -3.6845059, 3.6994267
9: 3.9322391, 6.5279784, 3.9070263, 6.5659070, -2.6336679, 2.6209521

Time for backsubstitution: 14.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 494

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5798

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8020336, upper bound: 1.8314892
time: 5.80 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053082, upper bound: 1.8314906
time: 5.06 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -9.7020168, -5.1795983, -9.6324787, -5.2194176, -3.9592133, 4.0596552
1: -17.0970554, -13.3739710, -17.0599537, -13.3951693, -3.7018862, 3.6859827
2: -8.1175184, -4.2952156, -8.0496616, -4.3364553, -3.4148111, 3.5052404
3: -13.7561626, -8.7565775, -13.7152596, -8.7844372, -4.9717255, 4.9586821
4: -3.8358641, -0.2915483, -3.8215551, -0.3094628, -3.5264013, 3.5300069
5: -13.9660082, -10.0149679, -13.9259109, -10.0381699, -3.2709875, 3.2850437
6: -15.9357290, -11.4394979, -15.8907461, -11.4718838, -4.0318108, 4.0987720
7: -8.3226089, -4.2791243, -8.2959061, -4.3132200, -4.0093889, 4.0167818
8: -6.6781731, -3.0312710, -6.6310477, -3.0642805, -3.6138926, 3.5997767
9: 3.9826174, 6.5122128, 4.0080853, 6.4742885, -2.4916711, 2.5041275

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5798

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8020323, upper bound: 1.8318860
time: 4.46 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053093, upper bound: 1.8318858
time: 4.46 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -9.7020168, -5.1795983, -9.6516857, -5.2123361, -3.9785881, 4.0788002
1: -17.0970554, -13.3739710, -17.0863495, -13.3565950, -3.7404604, 3.7123785
2: -8.1175184, -4.2952156, -8.1016855, -4.3077364, -3.4444532, 3.5496016
3: -13.7561626, -8.7565775, -13.7873831, -8.7640629, -4.9920998, 5.0308056
4: -3.8358641, -0.2915483, -3.8931284, -0.2618936, -3.5527773, 3.6015801
5: -13.9660082, -10.0149679, -13.9539986, -9.9989195, -3.3075628, 3.3114114
6: -15.9357290, -11.4394979, -15.9145517, -11.4365759, -4.0709410, 4.1284628
7: -8.3226089, -4.2791243, -8.3611193, -4.2134285, -4.1091805, 4.0819950
8: -6.6781731, -3.0312710, -6.6736207, -3.0213208, -3.6568522, 3.6423497
9: 3.9826174, 6.5122128, 3.9322443, 6.5279779, -2.5453606, 2.5799685

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5798

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8020323, upper bound: 1.8318861
time: 5.18 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053094, upper bound: 1.8318860
time: 5.05 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9.7212629, -5.1725130, -9.6324787, -5.2194176, -3.9805460, 4.0790615
1: -17.1234303, -13.3353443, -17.0599537, -13.3951693, -3.7282610, 3.7246094
2: -8.1695328, -4.2664700, -8.0496616, -4.3364553, -3.4585485, 3.5345764
3: -13.8282261, -8.7362127, -13.7152596, -8.7844372, -5.0437889, 4.9790468
4: -3.9074385, -0.2440178, -3.8215551, -0.3094628, -3.5979757, 3.5438771
5: -13.9940996, -9.9756966, -13.9259109, -10.0381699, -3.2973413, 3.3198705
6: -15.9595490, -11.4040546, -15.8907461, -11.4718838, -4.0614901, 4.1466050
7: -8.3881712, -4.1793394, -8.2959061, -4.3132200, -4.0749512, 4.1165667
8: -6.7207527, -2.9882340, -6.6310477, -3.0642805, -3.6564722, 3.6428137
9: 3.9066682, 6.5659094, 4.0080853, 6.4742885, -2.5676203, 2.5578241

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5798

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8020282, upper bound: 1.8580982
time: 4.53 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053053, upper bound: 1.8580978
time: 4.87 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.7212629, -5.1725130, -9.6517410, -5.2123356, -3.9978266, 4.0983448
1: -17.1234303, -13.3353443, -17.0863514, -13.3565893, -3.7668409, 3.7510071
2: -8.1695328, -4.2664700, -8.1016903, -4.3077345, -3.4719772, 3.5623574
3: -13.8282261, -8.7362127, -13.7873850, -8.7640591, -5.0641670, 5.0511723
4: -3.9074385, -0.2440178, -3.8931270, -0.2618673, -3.5960603, 3.5903630
5: -13.9940996, -9.9756966, -13.9539986, -9.9989147, -3.3251138, 3.3393126
6: -15.9595490, -11.4040546, -15.9145527, -11.4365282, -4.1295204, 4.1966419
7: -8.3881712, -4.1793394, -8.3611259, -4.2134304, -4.1747408, 4.1817865
8: -6.7207527, -2.9882340, -6.6736231, -3.0213170, -3.6994357, 3.6853890
9: 3.9066682, 6.5659094, 3.9322391, 6.5279784, -2.6213102, 2.6336703

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5798

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8020282, upper bound: 1.8581002
time: 5.25 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053053, upper bound: 1.8580995
time: 5.10 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.7020168, -5.1795983, -9.7020168, -5.1795983, -4.1246758, 4.1246758
1: -17.0970554, -13.3739710, -17.0970554, -13.3739710, -3.7230844, 3.7230844
2: -8.1175184, -4.2952156, -8.1175184, -4.2952156, -3.5802689, 3.5802693
3: -13.7561626, -8.7565775, -13.7561626, -8.7565775, -4.9995852, 4.9995852
4: -3.8358641, -0.2915483, -3.8358641, -0.2915483, -3.5443158, 3.5443158
5: -13.9660082, -10.0149679, -13.9660082, -10.0149679, -3.3205366, 3.3205366
6: -15.9357290, -11.4394979, -15.9357290, -11.4394979, -4.1388874, 4.1388879
7: -8.3226089, -4.2791243, -8.3226089, -4.2791243, -4.0434847, 4.0434847
8: -6.6781731, -3.0312710, -6.6781731, -3.0312710, -3.6469021, 3.6469021
9: 3.9826174, 6.5122128, 3.9826174, 6.5122128, -2.5295954, 2.5295954

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5798

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8033156, upper bound: 1.8053057
time: 4.66 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8065960, upper bound: 1.8341395
time: 4.35 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 29.79 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 29.79
Output dim: 9, lower bound: -1.8020354, upper bound: 1.8053095
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 29.79
Output dim: 9, lower bound: -1.8053124, upper bound: 1.8053092
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 29.79
Output dim: 9, lower bound: -1.8020353, upper bound: 1.8053095
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 29.79
Output dim: 9, lower bound: -1.8053123, upper bound: 1.8053100
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 29.79
Output dim: 9, lower bound: -1.8020313, upper bound: 1.8314888
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 29.79
Output dim: 9, lower bound: -1.8053083, upper bound: 1.8314885
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 29.79
Output dim: 9, lower bound: -1.8020312, upper bound: 1.8314887
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 29.79
Output dim: 9, lower bound: -1.8053083, upper bound: 1.8314894
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 29.79
Output dim: 9, lower bound: -1.8282759, upper bound: 1.8053047
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 29.79
Output dim: 9, lower bound: -1.8314922, upper bound: 1.8053055
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 29.79
Output dim: 9, lower bound: -1.8282754, upper bound: 1.8053049
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 29.79
Output dim: 9, lower bound: -1.8314916, upper bound: 1.8053049
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 29.79
Output dim: 9, lower bound: -1.8020334, upper bound: 1.8314905
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 29.79
Output dim: 9, lower bound: -1.8053082, upper bound: 1.8314898
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 29.79
Output dim: 9, lower bound: -1.8020336, upper bound: 1.8314892
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 29.79
Output dim: 9, lower bound: -1.8053082, upper bound: 1.8314906
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 29.79
Output dim: 9, lower bound: -1.8020323, upper bound: 1.8318860
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 29.79
Output dim: 9, lower bound: -1.8053093, upper bound: 1.8318858
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 29.79
Output dim: 9, lower bound: -1.8020323, upper bound: 1.8318861
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 29.79
Output dim: 9, lower bound: -1.8053094, upper bound: 1.8318860
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 29.79
Output dim: 9, lower bound: -1.8020282, upper bound: 1.8580982
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 29.79
Output dim: 9, lower bound: -1.8053053, upper bound: 1.8580978
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 29.79
Output dim: 9, lower bound: -1.8020282, upper bound: 1.8581002
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 29.79
Output dim: 9, lower bound: -1.8053053, upper bound: 1.8580995
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 29.79
Output dim: 9, lower bound: -1.8033156, upper bound: 1.8053057
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 29.79
Output dim: 9, lower bound: -1.8065960, upper bound: 1.8341395
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.79
Output dim: 9, lower bound: -1.8065950, upper bound: 1.8341431
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.79
Output dim: 9, lower bound: -1.8065950, upper bound: 1.8603826
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.79
Output dim: 9, lower bound: -1.8065950, upper bound: 1.8603806
Binary search (step 0): status=Status.UNKNOWN, k_low=6, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=2.6592936515808105
rel_dist={9: [-1.860453671254131, 1.860453011622619]}

## Binary search (step 1) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5875
type: A, layer: 1, pos: 4557
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5875

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7014879, upper bound: 1.6806596
time: 16.03 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7023552, upper bound: 1.7023542
time: 6.80 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 23.00 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 23.00
Output dim: 9, lower bound: -1.7014879, upper bound: 1.6806596
IS_A2, status: Status.UNKNOWN, split count: 1, time: 23.00
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

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4557
type: B, layer: 1, pos: 5875
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 5820
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4557

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6798732, upper bound: 1.6806059
time: 4.23 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7014325, upper bound: 1.6806059
time: 5.24 seconds

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

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4557
type: B, layer: 1, pos: 5875
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 5820
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4557

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6807020, upper bound: 1.7023002
time: 4.83 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7022996, upper bound: 1.7022989
time: 7.46 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 26.98 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 26.98
Output dim: 9, lower bound: -1.6798732, upper bound: 1.6806059
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 26.98
Output dim: 9, lower bound: -1.7014325, upper bound: 1.6806059
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 26.98
Output dim: 9, lower bound: -1.6807020, upper bound: 1.7023002
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 26.98
Output dim: 9, lower bound: -1.7022996, upper bound: 1.7022989

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

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4557
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4557

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6798731, upper bound: 1.6590683
time: 4.36 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6798731, upper bound: 1.6806059
time: 4.40 seconds

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

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4557
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4557

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7014331, upper bound: 1.6590679
time: 7.72 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7014332, upper bound: 1.6806050
time: 6.74 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -9.7162743, -5.1738381, -9.7020197, -5.1795998, -3.8553295, 3.8656149
1: -17.1165142, -13.3432341, -17.0970631, -13.3739748, -3.5836143, 3.5979185
2: -8.1543026, -4.2707682, -8.1175299, -4.2952166, -3.4178457, 3.4157748
3: -13.8075008, -8.7393017, -13.7561617, -8.7565784, -5.0509224, 5.0168600
4: -3.8878503, -0.2525269, -3.8358631, -0.2915449, -3.3995156, 3.3240957
5: -13.9895229, -9.9860983, -13.9660177, -10.0149670, -3.0595350, 3.0688720
6: -15.9552279, -11.4144745, -15.9357309, -11.4394970, -3.9033585, 3.9123812
7: -8.3728075, -4.2054820, -8.3226089, -4.2791204, -4.0885353, 4.0892630
8: -6.7133470, -3.0000010, -6.6781740, -3.0312705, -3.6820765, 3.6781731
9: 3.9189930, 6.5510359, 3.9826155, 6.5122166, -2.5932236, 2.5684204

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4557
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4557

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6807020, upper bound: 1.6807028
time: 4.91 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6807020, upper bound: 1.7023014
time: 4.98 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -9.7212772, -5.1725125, -9.7212658, -5.1725121, -3.8793869, 3.8838744
1: -17.1234398, -13.3353319, -17.1234379, -13.3353443, -3.6267538, 3.6313748
2: -8.1695595, -4.2664623, -8.1695414, -4.2664680, -3.4623489, 3.4425902
3: -13.8282642, -8.7362051, -13.8282318, -8.7362146, -5.0920496, 5.0920267
4: -3.9074695, -0.2440007, -3.9074397, -0.2440169, -3.4036732, 3.3736839
5: -13.9941082, -9.9756899, -13.9941053, -9.9756956, -3.0905685, 3.1043072
6: -15.9595528, -11.4040365, -15.9595518, -11.4040537, -3.9722795, 3.9602656
7: -8.3881950, -4.1793032, -8.3881712, -4.1793313, -4.1968775, 4.2012510
8: -6.7207656, -2.9882154, -6.7207580, -2.9882345, -3.7325311, 3.7325425
9: 3.9066486, 6.5659370, 3.9066677, 6.5659108, -2.6592622, 2.6592693

Time for backsubstitution: 14.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4557
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4557

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7023002, upper bound: 1.6807007
time: 5.16 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7023002, upper bound: 1.7022995
time: 4.93 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 24.76 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 24.76
Output dim: 9, lower bound: -1.6798731, upper bound: 1.6590683
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 24.76
Output dim: 9, lower bound: -1.6798731, upper bound: 1.6806059
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 24.76
Output dim: 9, lower bound: -1.7014331, upper bound: 1.6590679
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 24.76
Output dim: 9, lower bound: -1.7014332, upper bound: 1.6806050
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 24.76
Output dim: 9, lower bound: -1.6807020, upper bound: 1.6807028
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 24.76
Output dim: 9, lower bound: -1.6807020, upper bound: 1.7023014
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 24.76
Output dim: 9, lower bound: -1.7023002, upper bound: 1.6807007
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 24.76
Output dim: 9, lower bound: -1.7023002, upper bound: 1.7022995

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

Time for backsubstitution: 14.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5875
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 5820
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5875

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6592171, upper bound: 1.6590727
time: 4.85 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6592171, upper bound: 1.6590729
time: 5.16 seconds

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

Time for backsubstitution: 14.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5875
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 5820
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5875

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6592171, upper bound: 1.6806059
time: 4.40 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6592170, upper bound: 1.6806059
time: 5.55 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -9.6324787, -5.2194176, -9.7007294, -5.1739626, -3.6633835, 3.6733727
1: -17.0599537, -13.3951693, -17.1128464, -13.3359604, -3.5333529, 3.5343380
2: -8.0496616, -4.3364553, -8.1495161, -4.2671928, -3.2168064, 3.2388265
3: -13.7152596, -8.7844372, -13.8163595, -8.7373018, -4.9779577, 5.0319223
4: -3.8215551, -0.3094628, -3.9067566, -0.2491059, -3.3051286, 3.3933425
5: -13.9259109, -10.0381699, -13.9822283, -9.9761963, -2.9998026, 2.9970860
6: -15.8907461, -11.4718838, -15.9461765, -11.4057550, -3.8016701, 3.7837510
7: -8.2959061, -4.3132200, -8.3873749, -4.1892715, -4.0234947, 4.0173187
8: -6.6310477, -3.0642805, -6.7068405, -2.9896517, -3.6413960, 3.6425600
9: 4.0080853, 6.4742885, 3.9073792, 6.5545888, -2.5465035, 2.5669093

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5875
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 5820
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5875

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6592171, upper bound: 1.6590683
time: 4.58 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6592171, upper bound: 1.6590682
time: 4.68 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -9.6517410, -5.2123356, -9.7007294, -5.1739626, -3.6820078, 3.6928043
1: -17.0863514, -13.3565893, -17.1128464, -13.3359604, -3.5606432, 3.5681720
2: -8.1016903, -4.3077345, -8.1495161, -4.2671928, -3.2391863, 3.2464843
3: -13.7873850, -8.7640591, -13.8163595, -8.7373018, -5.0500832, 5.0523005
4: -3.8931270, -0.2618673, -3.9067566, -0.2491059, -3.3489132, 3.3513870
5: -13.9539986, -9.9989147, -13.9822283, -9.9761963, -3.0182834, 3.0243559
6: -15.9145527, -11.4365282, -15.9461765, -11.4057550, -3.8487682, 3.8494802
7: -8.3611259, -4.2134304, -8.3873749, -4.1892715, -4.1078358, 4.1090689
8: -6.6736231, -3.0213170, -6.7068405, -2.9896517, -3.6839714, 3.6855235
9: 3.9322391, 6.5279784, 3.9073792, 6.5545888, -2.6223497, 2.6205993

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5875
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 5820
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5875

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6592170, upper bound: 1.6806058
time: 4.99 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6592170, upper bound: 1.6806059
time: 5.26 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -9.7020168, -5.1795983, -9.7020197, -5.1795998, -3.8408775, 3.8426332
1: -17.0970554, -13.3739710, -17.0970631, -13.3739748, -3.5625095, 3.5686574
2: -8.1175184, -4.2952156, -8.1175299, -4.2952166, -3.3856993, 3.3907509
3: -13.7561626, -8.7565775, -13.7561617, -8.7565784, -4.9995842, 4.9995842
4: -3.8358641, -0.2915483, -3.8358631, -0.2915449, -3.3565087, 3.3495350
5: -13.9660082, -10.0149679, -13.9660177, -10.0149670, -3.0367727, 3.0426583
6: -15.9357290, -11.4394979, -15.9357309, -11.4394970, -3.8766842, 3.8821673
7: -8.3226089, -4.2791243, -8.3226089, -4.2791204, -4.0252352, 4.0205603
8: -6.6781731, -3.0312710, -6.6781740, -3.0312705, -3.6469026, 3.6469030
9: 3.9826174, 6.5122128, 3.9826155, 6.5122166, -2.5295992, 2.5295973

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5875
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 5820
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5875

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6590683, upper bound: 1.6798768
time: 4.39 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6590682, upper bound: 1.6807064
time: 4.45 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -9.7212019, -5.1725130, -9.7020197, -5.1795998, -3.8619614, 3.8651593
1: -17.1234283, -13.3353481, -17.0970631, -13.3739748, -3.5916204, 3.6040912
2: -8.1695290, -4.2664680, -8.1175299, -4.2952166, -3.4325342, 3.4162509
3: -13.8282280, -8.7362137, -13.7561617, -8.7565784, -5.0716496, 5.0199480
4: -3.9074385, -0.2440448, -3.8358631, -0.2915449, -3.4138007, 3.3325844
5: -13.9941006, -9.9757042, -13.9660177, -10.0149670, -3.0631609, 3.0757575
6: -15.9595480, -11.4041023, -15.9357309, -11.4394970, -3.9063931, 3.9215236
7: -8.3881655, -4.1793394, -8.3226089, -4.2791204, -4.1029987, 4.1102347
8: -6.7207499, -2.9882340, -6.6781740, -3.0312705, -3.6894794, 3.6899400
9: 3.9066725, 6.5659094, 3.9826155, 6.5122166, -2.6055441, 2.5832939

Time for backsubstitution: 14.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5875
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 5820
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5875

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6590682, upper bound: 1.7014322
time: 4.58 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6590682, upper bound: 1.7023014
time: 4.38 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -9.7020168, -5.1795983, -9.7212658, -5.1725121, -3.8622074, 3.8604662
1: -17.0970554, -13.3739710, -17.1234379, -13.3353443, -3.5979462, 3.5974550
2: -8.1175184, -4.2952156, -8.1695414, -4.2664680, -3.4150572, 3.4224091
3: -13.7561626, -8.7565775, -13.8282318, -8.7362146, -5.0199480, 5.0716543
4: -3.8358641, -0.2915483, -3.9074397, -0.2440169, -3.3368101, 3.4158487
5: -13.9660082, -10.0149679, -13.9941053, -9.9756956, -3.0720215, 3.0690494
6: -15.9357290, -11.4394979, -15.9595518, -11.4040537, -3.9251204, 3.9118729
7: -8.3226089, -4.2791243, -8.3881712, -4.1793313, -4.0998926, 4.1004210
8: -6.6781731, -3.0312710, -6.7207580, -2.9882345, -3.6899385, 3.6894870
9: 3.9826174, 6.5122128, 3.9066677, 6.5659108, -2.5832934, 2.6055450

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5875
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 5820
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5875

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6590683, upper bound: 1.6798723
time: 4.79 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6590682, upper bound: 1.6807039
time: 4.59 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.7212629, -5.1725130, -9.7212658, -5.1725121, -3.8807592, 3.8838696
1: -17.1234303, -13.3353443, -17.1234379, -13.3353443, -3.6252146, 3.6313629
2: -8.1695328, -4.2664700, -8.1695414, -4.2664680, -3.4375315, 3.4425831
3: -13.8282261, -8.7362127, -13.8282318, -8.7362146, -5.0920115, 5.0920191
4: -3.9074385, -0.2440178, -3.9074397, -0.2440169, -3.3806400, 3.3736668
5: -13.9940996, -9.9756966, -13.9941053, -9.9756956, -3.0905609, 3.0964465
6: -15.9595490, -11.4040546, -15.9595518, -11.4040537, -3.9722528, 3.9777336
7: -8.3881712, -4.1793394, -8.3881712, -4.1793313, -4.1968555, 4.1921797
8: -6.7207527, -2.9882340, -6.7207580, -2.9882345, -3.7325182, 3.7325239
9: 3.9066682, 6.5659094, 3.9066677, 6.5659108, -2.6592426, 2.6592417

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5875
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 5820
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5875

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6590682, upper bound: 1.7014322
time: 5.21 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6590682, upper bound: 1.6807016
time: 4.94 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 24.84 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.84
Output dim: 9, lower bound: -1.6592171, upper bound: 1.6590727
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.84
Output dim: 9, lower bound: -1.6592171, upper bound: 1.6590729
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.84
Output dim: 9, lower bound: -1.6592171, upper bound: 1.6806059
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.84
Output dim: 9, lower bound: -1.6592170, upper bound: 1.6806059
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.84
Output dim: 9, lower bound: -1.6592171, upper bound: 1.6590683
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.84
Output dim: 9, lower bound: -1.6592171, upper bound: 1.6590682
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.84
Output dim: 9, lower bound: -1.6592170, upper bound: 1.6806058
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.84
Output dim: 9, lower bound: -1.6592170, upper bound: 1.6806059
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.84
Output dim: 9, lower bound: -1.6590683, upper bound: 1.6798768
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.84
Output dim: 9, lower bound: -1.6590682, upper bound: 1.6807064
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.84
Output dim: 9, lower bound: -1.6590682, upper bound: 1.7014322
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.84
Output dim: 9, lower bound: -1.6590682, upper bound: 1.7023014
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.84
Output dim: 9, lower bound: -1.6590683, upper bound: 1.6798723
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.84
Output dim: 9, lower bound: -1.6590682, upper bound: 1.6807039
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.84
Output dim: 9, lower bound: -1.6590682, upper bound: 1.7014322
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.84
Output dim: 9, lower bound: -1.6590682, upper bound: 1.6807016

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

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5798

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6564310, upper bound: 1.6590705
time: 4.34 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6592191, upper bound: 1.6590704
time: 4.38 seconds

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

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5798

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6564310, upper bound: 1.6590700
time: 4.90 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6592191, upper bound: 1.6590704
time: 4.91 seconds

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

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5798

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6564265, upper bound: 1.6806021
time: 4.49 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6592146, upper bound: 1.6806021
time: 4.78 seconds

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

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5798

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6564265, upper bound: 1.6806018
time: 5.77 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6592145, upper bound: 1.6806021
time: 5.18 seconds

## BFS IS instance: IS_A1_B2_A1_B1

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

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5798

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6779887, upper bound: 1.6590660
time: 4.53 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6807501, upper bound: 1.6590662
time: 4.98 seconds

## BFS IS instance: IS_A1_B2_A1_B2

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

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5798

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6779887, upper bound: 1.6590657
time: 6.87 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6807501, upper bound: 1.6590658
time: 4.55 seconds

## BFS IS instance: IS_A1_B2_A2_B1

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

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5798

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6566219, upper bound: 1.6806031
time: 5.31 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6594181, upper bound: 1.6806034
time: 4.78 seconds

## BFS IS instance: IS_A1_B2_A2_B2

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

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5798

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6566219, upper bound: 1.6806030
time: 7.12 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6594181, upper bound: 1.6806031
time: 7.08 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -9.7020168, -5.1795983, -9.6324787, -5.2194176, -3.6737041, 3.7724423
1: -17.0970554, -13.3739710, -17.0599537, -13.3951693, -3.5165854, 3.5307760
2: -8.1175184, -4.2952156, -8.0496616, -4.3364553, -3.2128682, 3.3161459
3: -13.7561626, -8.7565775, -13.7152596, -8.7844372, -4.9717255, 4.9586821
4: -3.8358641, -0.2915483, -3.8215551, -0.3094628, -3.3378506, 3.3311191
5: -13.9660082, -10.0149679, -13.9259109, -10.0381699, -2.9827776, 3.0019712
6: -15.9357290, -11.4394979, -15.8907461, -11.4718838, -3.7674923, 3.8372145
7: -8.3226089, -4.2791243, -8.2959061, -4.3132200, -3.9840498, 3.9466047
8: -6.6781731, -3.0312710, -6.6310477, -3.0642805, -3.6138926, 3.5997767
9: 3.9826174, 6.5122128, 4.0080853, 6.4742885, -2.4916711, 2.5041275

Time for backsubstitution: 14.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5798

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6562822, upper bound: 1.6798742
time: 4.41 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6590703, upper bound: 1.6798742
time: 4.46 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -9.7020168, -5.1795983, -9.7020168, -5.1795983, -3.8398905, 3.8398907
1: -17.0970554, -13.3739710, -17.0970554, -13.3739710, -3.5625086, 3.5625086
2: -8.1175184, -4.2952156, -8.1175184, -4.2952156, -3.3907452, 3.3907447
3: -13.7561626, -8.7565775, -13.7561626, -8.7565775, -4.9995852, 4.9995852
4: -3.8358641, -0.2915483, -3.8358641, -0.2915483, -3.3495345, 3.3495340
5: -13.9660082, -10.0149679, -13.9660082, -10.0149679, -3.0367727, 3.0367723
6: -15.9357290, -11.4394979, -15.9357290, -11.4394979, -3.8766851, 3.8766861
7: -8.3226089, -4.2791243, -8.3226089, -4.2791243, -4.0252314, 4.0252314
8: -6.6781731, -3.0312710, -6.6781731, -3.0312710, -3.6469021, 3.6469021
9: 3.9826174, 6.5122128, 3.9826174, 6.5122128, -2.5295954, 2.5295954

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5798

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6562822, upper bound: 1.6807041
time: 4.55 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6590702, upper bound: 1.6807038
time: 4.52 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9.7212019, -5.1725130, -9.6324787, -5.2194176, -3.6924133, 3.7950385
1: -17.1234283, -13.3353481, -17.0599537, -13.3951693, -3.5456977, 3.5662103
2: -8.1695290, -4.2664680, -8.0496616, -4.3364553, -3.2398643, 3.3436170
3: -13.8282280, -8.7362137, -13.7152596, -8.7844372, -5.0437908, 4.9790459
4: -3.9074385, -0.2440448, -3.8215551, -0.3094628, -3.3951387, 3.3143477
5: -13.9941006, -9.9757042, -13.9259109, -10.0381699, -3.0091295, 3.0350854
6: -15.9595480, -11.4041023, -15.8907461, -11.4718838, -3.7971706, 3.8765707
7: -8.3881655, -4.1793394, -8.2959061, -4.3132200, -4.0619669, 4.0253983
8: -6.7207499, -2.9882340, -6.6310477, -3.0642805, -3.6564693, 3.6428137
9: 3.9066725, 6.5659094, 4.0080853, 6.4742885, -2.5676160, 2.5578241

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5798

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6562777, upper bound: 1.7014285
time: 4.77 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6590657, upper bound: 1.7014285
time: 4.47 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.7212019, -5.1725130, -9.7020168, -5.1795983, -3.8583555, 3.8622079
1: -17.1234283, -13.3353481, -17.0970554, -13.3739710, -3.5916195, 3.5979443
2: -8.1695290, -4.2664680, -8.1175184, -4.2952156, -3.4224057, 3.4162502
3: -13.8282280, -8.7362137, -13.7561626, -8.7565775, -5.0716505, 5.0199490
4: -3.9074385, -0.2440448, -3.8358641, -0.2915483, -3.4105525, 3.3325839
5: -13.9941006, -9.9757042, -13.9660082, -10.0149679, -3.0631609, 3.0730186
6: -15.9595480, -11.4041023, -15.9357290, -11.4394979, -3.9063931, 3.9160419
7: -8.3881655, -4.1793394, -8.3226089, -4.2791243, -4.1029930, 4.0998869
8: -6.7207499, -2.9882340, -6.6781731, -3.0312710, -3.6894789, 3.6899390
9: 3.9066725, 6.5659094, 3.9826174, 6.5122128, -2.6055403, 2.5832920

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5798

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6562777, upper bound: 1.7022968
time: 4.58 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6590657, upper bound: 1.7022986
time: 5.11 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.7020168, -5.1795983, -9.6517410, -5.2123356, -3.6950321, 3.7902861
1: -17.0970554, -13.3739710, -17.0863514, -13.3565893, -3.5519490, 3.5595913
2: -8.1175184, -4.2952156, -8.1016903, -4.3077345, -3.2338357, 3.3497653
3: -13.7561626, -8.7565775, -13.7873850, -8.7640591, -4.9921036, 5.0308075
4: -3.8358641, -0.2915483, -3.8931270, -0.2618673, -3.3182030, 3.3910956
5: -13.9660082, -10.0149679, -13.9539986, -9.9989147, -3.0146742, 3.0283403
6: -15.9357290, -11.4394979, -15.9145527, -11.4365282, -3.8157845, 3.8669043
7: -8.3226089, -4.2791243, -8.3611259, -4.2134304, -4.0605960, 4.0263386
8: -6.6781731, -3.0312710, -6.6736231, -3.0213170, -3.6568561, 3.6423521
9: 3.9826174, 6.5122128, 3.9322391, 6.5279784, -2.5453610, 2.5799737

Time for backsubstitution: 14.56 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=6, k_high=8, k_mid=7, eps_mid=0.0273438, abs_max=2.6592936515808105
rel_dist={9: [-1.702359651903449, 1.7023598330847598]}

## Binary search (step 2) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5875
type: A, layer: 1, pos: 4557
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5875

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6174309, upper bound: 1.5990262
time: 4.85 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6176209, upper bound: 1.6176201
time: 4.99 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 10.01 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 10.01
Output dim: 9, lower bound: -1.6174309, upper bound: 1.5990262
IS_A2, status: Status.UNKNOWN, split count: 1, time: 10.01
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

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4557
type: B, layer: 1, pos: 5875
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 5820
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4557

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5981700, upper bound: 1.5989753
time: 5.46 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6173802, upper bound: 1.5989754
time: 7.32 seconds

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

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4557
type: B, layer: 1, pos: 5875
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 5820
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4557

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5982997, upper bound: 1.6175683
time: 4.84 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6175701, upper bound: 1.6175700
time: 5.36 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 24.91 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 24.91
Output dim: 9, lower bound: -1.5981700, upper bound: 1.5989753
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 24.91
Output dim: 9, lower bound: -1.6173802, upper bound: 1.5989754
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 24.91
Output dim: 9, lower bound: -1.5982997, upper bound: 1.6175683
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 24.91
Output dim: 9, lower bound: -1.6175701, upper bound: 1.6175700

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

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4557
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 494

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4557

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6173807, upper bound: 1.5797849
time: 7.76 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6173807, upper bound: 1.5989750
time: 6.91 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -9.7153816, -5.1740894, -9.7020216, -5.1795979, -3.7121925, 3.7226479
1: -17.1152840, -13.3447189, -17.0970631, -13.3739719, -3.4509974, 3.4659152
2: -8.1515255, -4.2715635, -8.1175289, -4.2952137, -3.3192673, 3.3201857
3: -13.8037395, -8.7398682, -13.7561588, -8.7565813, -4.9994450, 5.0162907
4: -3.8842866, -0.2540438, -3.8358629, -0.2915471, -3.2769513, 3.2061725
5: -13.9886570, -9.9879856, -13.9660149, -10.0149679, -2.9169726, 2.9254804
6: -15.9544201, -11.4163418, -15.9357319, -11.4394960, -3.7710075, 3.7799683
7: -8.3699265, -4.2102299, -8.3226118, -4.2791190, -3.9755573, 3.9714260
8: -6.7119670, -3.0021410, -6.6781750, -3.0312700, -3.6806970, 3.6760340
9: 3.9212894, 6.5483236, 3.9826169, 6.5122147, -2.5909252, 2.5657067

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4557
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4557

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5982997, upper bound: 1.5982986
time: 4.79 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5982997, upper bound: 1.6175684
time: 4.75 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -9.7212772, -5.1725125, -9.7212648, -5.1725121, -3.7383862, 3.7408371
1: -17.1234398, -13.3353319, -17.1234341, -13.3353462, -3.4955778, 3.5004697
2: -8.1695595, -4.2664623, -8.1695433, -4.2664685, -3.3664474, 3.3451853
3: -13.8282642, -8.7362051, -13.8282309, -8.7362137, -5.0469990, 5.0548840
4: -3.9074695, -0.2440007, -3.9074388, -0.2440171, -3.2880087, 3.2562585
5: -13.9941082, -9.9756899, -13.9941025, -9.9756956, -2.9484353, 2.9629822
6: -15.9595528, -11.4040365, -15.9595509, -11.4040546, -3.8400297, 3.8297858
7: -8.3881950, -4.1793032, -8.3881702, -4.1793318, -4.0859280, 4.0908747
8: -6.7207656, -2.9882154, -6.7207575, -2.9882340, -3.7325315, 3.7325420
9: 3.9066486, 6.5659370, 3.9066677, 6.5659094, -2.6592607, 2.6592693

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4557
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4557

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6175691, upper bound: 1.5982991
time: 5.58 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6175693, upper bound: 1.5982987
time: 5.27 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 25.56 seconds
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 25.56
Output dim: 9, lower bound: -1.6173807, upper bound: 1.5797849
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 25.56
Output dim: 9, lower bound: -1.6173807, upper bound: 1.5989750
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 25.56
Output dim: 9, lower bound: -1.5982997, upper bound: 1.5982986
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 25.56
Output dim: 9, lower bound: -1.5982997, upper bound: 1.6175684
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 25.56
Output dim: 9, lower bound: -1.6175691, upper bound: 1.5982991
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 25.56
Output dim: 9, lower bound: -1.6175693, upper bound: 1.5982987

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -9.6324787, -5.2194176, -9.6966047, -5.1739979, -3.5215683, 3.5260315
1: -17.0599537, -13.3951693, -17.1107254, -13.3360157, -3.4011192, 3.3998775
2: -8.0496616, -4.3364553, -8.1454897, -4.2672081, -3.1167173, 3.1292789
3: -13.7152596, -8.7844372, -13.8139782, -8.7374249, -4.9110804, 4.9257460
4: -3.8215551, -0.3094628, -3.9066582, -0.2501222, -3.1877971, 3.2726538
5: -13.9259109, -10.0381699, -13.9798422, -9.9762096, -2.8558969, 2.8505569
6: -15.8907461, -11.4718838, -15.9434900, -11.4059124, -3.6696310, 3.6488972
7: -8.2959061, -4.3132200, -8.3872986, -4.1912642, -3.9012184, 3.9048805
8: -6.6310477, -3.0642805, -6.7040415, -2.9897594, -3.6411076, 3.5920753
9: 4.0080853, 6.4742885, 3.9074507, 6.5523119, -2.5442266, 2.5668378

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5875
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 5820
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5875

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5805115, upper bound: 1.5797845
time: 9.13 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5805115, upper bound: 1.5797873
time: 7.09 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -9.6517410, -5.2123356, -9.6966047, -5.1739979, -3.5398426, 3.5465326
1: -17.0863514, -13.3565893, -17.1107254, -13.3360157, -3.4282618, 3.4336224
2: -8.1016903, -4.3077345, -8.1454897, -4.2672081, -3.1364460, 3.1397305
3: -13.7873850, -8.7640591, -13.8139782, -8.7374249, -4.9454288, 4.9454088
4: -3.8931270, -0.2618673, -3.9066582, -0.2501222, -3.2302308, 3.2337680
5: -13.9539986, -9.9989147, -13.9798422, -9.9762096, -2.8739157, 2.8775768
6: -15.9145527, -11.4365282, -15.9434900, -11.4059124, -3.7152786, 3.7134762
7: -8.3611259, -4.2134304, -8.3872986, -4.1912642, -3.9928865, 3.9960995
8: -6.6736231, -3.0213170, -6.7040415, -2.9897594, -3.6265001, 3.6246662
9: 3.9322391, 6.5279784, 3.9074507, 6.5523119, -2.6200728, 2.6205277

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5875
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 5820
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5875

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5805114, upper bound: 1.5797873
time: 6.11 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5805114, upper bound: 1.5797850
time: 7.95 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -9.7212019, -5.1725130, -9.7020216, -5.1795979, -3.7200642, 3.7224712
1: -17.1234283, -13.3353481, -17.0970631, -13.3739719, -3.4604263, 3.4733338
2: -8.1695290, -4.2664680, -8.1175289, -4.2952137, -3.3366203, 3.3189969
3: -13.8282280, -8.7362137, -13.7561588, -8.7565813, -5.0255985, 5.0178185
4: -3.9074385, -0.2440448, -3.8358629, -0.2915471, -3.2912440, 3.2162600
5: -13.9941006, -9.9757042, -13.9660149, -10.0149679, -2.9212775, 2.9303265
6: -15.9595480, -11.4041023, -15.9357319, -11.4394960, -3.7752914, 3.7907877
7: -8.3881655, -4.1793394, -8.3226118, -4.2791190, -3.9926348, 3.9902506
8: -6.7207499, -2.9882340, -6.6781750, -3.0312700, -3.6894798, 3.6899409
9: 3.9066725, 6.5659094, 3.9826169, 6.5122147, -2.6055422, 2.5832925

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5875
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 5820
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5875

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5797849, upper bound: 1.6173799
time: 5.23 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5797849, upper bound: 1.6175697
time: 5.36 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -9.7020168, -5.1795983, -9.7212648, -5.1725121, -3.7212839, 3.7160864
1: -17.0970554, -13.3739710, -17.1234341, -13.3353462, -3.4668283, 3.4666409
2: -8.1175184, -4.2952156, -8.1695433, -4.2664685, -3.3203526, 3.3222735
3: -13.7561626, -8.7565775, -13.8282309, -8.7362137, -5.0109539, 5.0352182
4: -3.8358641, -0.2915483, -3.9074388, -0.2440171, -3.2211466, 3.2921808
5: -13.9660082, -10.0149679, -13.9941025, -9.9756956, -2.9303498, 2.9275126
6: -15.9357290, -11.4394979, -15.9595509, -11.4040546, -3.7943182, 3.7810936
7: -8.3226089, -4.2791243, -8.3881702, -4.1793318, -3.9818153, 3.9900446
8: -6.6781731, -3.0312710, -6.7207575, -2.9882340, -3.6899390, 3.6894865
9: 3.9826174, 6.5122128, 3.9066677, 6.5659094, -2.5832920, 2.6055450

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5875
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 5820
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5875

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5797850, upper bound: 1.5981693
time: 6.70 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5797849, upper bound: 1.5983005
time: 8.05 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.7212629, -5.1725130, -9.7212648, -5.1725121, -3.7394876, 3.7408321
1: -17.1234303, -13.3353443, -17.1234341, -13.3353462, -3.4939499, 3.5004578
2: -8.1695328, -4.2664700, -8.1695433, -4.2664685, -3.3401747, 3.3451781
3: -13.8282261, -8.7362127, -13.8282309, -8.7362137, -5.0452614, 5.0548782
4: -3.9074385, -0.2440178, -3.9074388, -0.2440171, -3.2636251, 3.2562413
5: -13.9940996, -9.9756966, -13.9941025, -9.9756956, -2.9484277, 2.9546595
6: -15.9595490, -11.4040546, -15.9595509, -11.4040546, -3.8400030, 3.8458071
7: -8.3881712, -4.1793394, -8.3881702, -4.1793318, -4.0859060, 4.0812731
8: -6.7207527, -2.9882340, -6.7207575, -2.9882340, -3.7325187, 3.7325234
9: 3.9066682, 6.5659094, 3.9066677, 6.5659094, -2.6592412, 2.6592417

Time for backsubstitution: 14.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5875
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 5820
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5875

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5797849, upper bound: 1.6173795
time: 7.26 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5797849, upper bound: 1.5982996
time: 6.80 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 28.82 seconds
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 28.82
Output dim: 9, lower bound: -1.5805115, upper bound: 1.5797845
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 28.82
Output dim: 9, lower bound: -1.5805115, upper bound: 1.5797873
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 28.82
Output dim: 9, lower bound: -1.5805114, upper bound: 1.5797873
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 28.82
Output dim: 9, lower bound: -1.5805114, upper bound: 1.5797850
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.82
Output dim: 9, lower bound: -1.5797849, upper bound: 1.6173799
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.82
Output dim: 9, lower bound: -1.5797849, upper bound: 1.6175697
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 28.82
Output dim: 9, lower bound: -1.5797850, upper bound: 1.5981693
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 28.82
Output dim: 9, lower bound: -1.5797849, upper bound: 1.5983005
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.82
Output dim: 9, lower bound: -1.5797849, upper bound: 1.6173795
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 28.82
Output dim: 9, lower bound: -1.5797849, upper bound: 1.5982996

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9.7212019, -5.1725130, -9.6324787, -5.2194176, -3.5428276, 3.6523499
1: -17.1234283, -13.3353481, -17.0599537, -13.3951693, -3.4134393, 3.4354529
2: -8.1695290, -4.2664680, -8.0496616, -4.3364553, -3.1305237, 3.2465796
3: -13.8282280, -8.7362137, -13.7152596, -8.7844372, -4.9401093, 4.9767523
4: -3.9074385, -0.2440448, -3.8215551, -0.3094628, -3.2725830, 3.1979589
5: -13.9941006, -9.9757042, -13.9259109, -10.0381699, -2.8649173, 2.8896537
6: -15.9595480, -11.4041023, -15.8907461, -11.4718838, -3.6650105, 3.7458353
7: -8.3881655, -4.1793394, -8.2959061, -4.3132200, -3.9518032, 3.9035006
8: -6.7207499, -2.9882340, -6.6310477, -3.0642805, -3.6099563, 3.6428137
9: 3.9066725, 6.5659094, 4.0080853, 6.4742885, -2.5676160, 2.5578241

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5798

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5772660, upper bound: 1.6173765
time: 4.99 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5797828, upper bound: 1.6173762
time: 5.22 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.7212019, -5.1725130, -9.7020168, -5.1795983, -3.7140675, 3.7193818
1: -17.1234283, -13.3353481, -17.0970554, -13.3739710, -3.4604254, 3.4668264
2: -8.1695290, -4.2664680, -8.1175184, -4.2952156, -3.3222699, 3.3189960
3: -13.8282280, -8.7362137, -13.7561626, -8.7565775, -5.0255985, 5.0082026
4: -3.9074385, -0.2440448, -3.8358641, -0.2915483, -3.2875857, 3.2162595
5: -13.9941006, -9.9757042, -13.9660082, -10.0149679, -2.9212780, 2.9272404
6: -15.9595480, -11.4041023, -15.9357290, -11.4394979, -3.7752914, 3.7849841
7: -8.3881655, -4.1793394, -8.3226089, -4.2791243, -3.9926291, 3.9818101
8: -6.7207499, -2.9882340, -6.6781731, -3.0312710, -3.6894789, 3.6899390
9: 3.9066725, 6.5659094, 3.9826174, 6.5122128, -2.6055403, 2.5832920

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5798

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5772659, upper bound: 1.6175661
time: 5.03 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5797828, upper bound: 1.6175660
time: 5.50 seconds

## BFS IS instance: IS_A2_B2_A2_B1

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

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5798

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5780859, upper bound: 1.6173772
time: 4.70 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5805175, upper bound: 1.6173777
time: 5.24 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 30.31 seconds
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 30.31
Output dim: 9, lower bound: -1.5772660, upper bound: 1.6173765
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 30.31
Output dim: 9, lower bound: -1.5797828, upper bound: 1.6173762
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 30.31
Output dim: 9, lower bound: -1.5772659, upper bound: 1.6175661
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 30.31
Output dim: 9, lower bound: -1.5797828, upper bound: 1.6175660
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 30.31
Output dim: 9, lower bound: -1.5780859, upper bound: 1.6173772
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 30.31
Output dim: 9, lower bound: -1.5805175, upper bound: 1.6173777

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -9.7140036, -5.1843190, -9.6318531, -5.2233381, -3.5286736, 3.6399298
1: -17.1194725, -13.3456497, -17.0591755, -13.3986721, -3.4055991, 3.4257164
2: -8.1602211, -4.2808328, -8.0483170, -4.3413010, -3.1130748, 3.2308002
3: -13.8088732, -8.7460575, -13.7091141, -8.7853661, -4.9197941, 4.9609890
4: -3.8913062, -0.2773978, -3.8211198, -0.3208202, -3.2349463, 3.1644607
5: -13.9653378, -9.9875774, -13.9160442, -10.0382595, -2.8358884, 2.8591175
6: -15.9417152, -11.4114227, -15.8850088, -11.4721937, -3.6469364, 3.7309027
7: -8.3680077, -4.1897736, -8.2901211, -4.3139377, -3.9310741, 3.8831062
8: -6.7104511, -2.9915304, -6.6279259, -3.0649648, -3.5996141, 3.6363955
9: 3.9149723, 6.5601497, 4.0106559, 6.4734430, -2.5584707, 2.5494938

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 5820
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 961

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5606078, upper bound: 1.6111412
time: 5.62 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5820

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5723718, upper bound: 1.6170497
time: 4.81 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5772601, upper bound: 1.6173708
time: 5.26 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -9.7212048, -5.1725135, -9.6324797, -5.2194195, -3.5385189, 3.6453381
1: -17.1234245, -13.3353519, -17.0599556, -13.3951702, -3.4122877, 3.4337463
2: -8.1695251, -4.2664709, -8.0496635, -4.3364568, -3.1265087, 3.2448311
3: -13.8282280, -8.7362194, -13.7152596, -8.7844353, -4.9272413, 4.9767532
4: -3.9074388, -0.2440495, -3.8215563, -0.3094621, -3.2571869, 3.1638002
5: -13.9940948, -9.9757042, -13.9259100, -10.0381727, -2.8383112, 2.8772368
6: -15.9595442, -11.4041052, -15.8907433, -11.4718838, -3.6536093, 3.7443390
7: -8.3881588, -4.1793408, -8.2959051, -4.3132186, -3.9400692, 3.8961797
8: -6.7207456, -2.9882364, -6.6310477, -3.0642810, -3.6181674, 3.6428113
9: 3.9066763, 6.5659075, 4.0080862, 6.4742880, -2.5676117, 2.5578213

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 5820
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 961

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5631248, upper bound: 1.6111409
time: 5.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5798

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5797829, upper bound: 1.6149449
time: 4.98 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5797829, upper bound: 1.6173763
time: 5.31 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -9.7140036, -5.1843190, -9.7013903, -5.1835184, -3.6999106, 3.7069616
1: -17.1194725, -13.3456497, -17.0962753, -13.3774776, -3.4525881, 3.4570885
2: -8.1602211, -4.2808328, -8.1161718, -4.3000579, -3.3047724, 3.3032146
3: -13.8088732, -8.7460575, -13.7500095, -8.7575140, -5.0052795, 4.9924355
4: -3.8913062, -0.2773978, -3.8354247, -0.3029032, -3.2499504, 3.1827679
5: -13.9653378, -9.9875774, -13.9561634, -10.0150604, -2.8922253, 2.8967099
6: -15.9417152, -11.4114227, -15.9299974, -11.4398117, -3.7572193, 3.7700520
7: -8.3680077, -4.1897736, -8.3168659, -4.2798424, -3.9719009, 3.9614558
8: -6.7104511, -2.9915304, -6.6750488, -3.0319538, -3.6784973, 3.6835184
9: 3.9149723, 6.5601497, 3.9851866, 6.5113692, -2.5963969, 2.5749631

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 5820
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 961

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5737442, upper bound: 1.6174498
time: 6.32 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5739933, upper bound: 1.6175560
time: 5.11 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.7212048, -5.1725135, -9.7020168, -5.1795993, -3.7097564, 3.7123673
1: -17.1234245, -13.3353519, -17.0970535, -13.3739719, -3.4592733, 3.4650936
2: -8.1695251, -4.2664709, -8.1175203, -4.2952151, -3.3182530, 3.3172479
3: -13.8282280, -8.7362194, -13.7561588, -8.7565794, -5.0127335, 5.0082026
4: -3.9074388, -0.2440495, -3.8358641, -0.2915480, -3.2721891, 3.1820879
5: -13.9940948, -9.9757042, -13.9660130, -10.0149651, -2.8945551, 2.9148233
6: -15.9595442, -11.4041052, -15.9357300, -11.4394989, -3.7638531, 3.7834883
7: -8.3881588, -4.1793408, -8.3226109, -4.2791262, -3.9808941, 3.9744883
8: -6.7207456, -2.9882364, -6.6781731, -3.0312710, -3.6894746, 3.6899366
9: 3.9066763, 6.5659075, 3.9826164, 6.5122132, -2.6055369, 2.5832911

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 5820
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 961

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5762617, upper bound: 1.6174499
time: 5.42 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5765102, upper bound: 1.6175576
time: 4.78 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -9.7140598, -5.1843195, -9.6511183, -5.2162519, -3.5523529, 3.6582875
1: -17.1194725, -13.3456459, -17.0855732, -13.3600903, -3.4390206, 3.4528589
2: -8.1602259, -4.2808299, -8.1003542, -4.3125753, -3.1383257, 3.2550187
3: -13.8088713, -8.7460546, -13.7811718, -8.7650099, -4.9394360, 4.9980564
4: -3.8913054, -0.2773726, -3.8926833, -0.2732220, -3.2154050, 3.2043900
5: -13.9653397, -9.9875698, -13.9441414, -9.9990101, -2.8629980, 2.8877439
6: -15.9417152, -11.4113770, -15.9088230, -11.4368458, -3.7115002, 3.7859287
7: -8.3680105, -4.1897740, -8.3554020, -4.2141457, -4.0243597, 3.9889345
8: -6.7104511, -2.9915299, -6.6705160, -3.0219989, -3.6321363, 3.6789861
9: 3.9149680, 6.5601501, 3.9348059, 6.5271311, -2.6121631, 2.6253443

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 5820
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 961

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5614506, upper bound: 1.6111416
time: 4.79 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5820

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5730927, upper bound: 1.6170486
time: 5.39 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5780802, upper bound: 1.6173714
time: 4.89 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -9.7212620, -5.1725149, -9.6517439, -5.2123361, -3.5621901, 3.6636982
1: -17.1234283, -13.3353519, -17.0863495, -13.3565884, -3.4457383, 3.4608746
2: -8.1695309, -4.2664738, -8.1016884, -4.3077345, -3.1517763, 3.2696161
3: -13.8282242, -8.7362146, -13.7873821, -8.7640610, -4.9469070, 5.0138521
4: -3.9074361, -0.2440253, -3.8931251, -0.2618670, -3.2435489, 3.2036810
5: -13.9940929, -9.9756966, -13.9539957, -9.9989119, -2.8652887, 2.9058867
6: -15.9595442, -11.4040546, -15.9145489, -11.4365292, -3.7181196, 3.7993603
7: -8.3881664, -4.1793404, -8.3611259, -4.2134275, -4.0333624, 4.0051680
8: -6.7207499, -2.9882360, -6.6736207, -3.0213213, -3.6506472, 3.6853848
9: 3.9066691, 6.5659080, 3.9322386, 6.5279779, -2.6213088, 2.6336694

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 5820
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 961

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5638822, upper bound: 1.6111422
time: 4.91 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5798

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5805177, upper bound: 1.6149465
time: 4.83 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5805175, upper bound: 1.6173776
time: 4.80 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 35.12 seconds
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 35.12
Output dim: 9, lower bound: -1.5723718, upper bound: 1.6170497
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 35.12
Output dim: 9, lower bound: -1.5772601, upper bound: 1.6173708
IS_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 35.12
Output dim: 9, lower bound: -1.5797829, upper bound: 1.6149449
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 35.12
Output dim: 9, lower bound: -1.5797829, upper bound: 1.6173763
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 35.12
Output dim: 9, lower bound: -1.5737442, upper bound: 1.6174498
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 35.12
Output dim: 9, lower bound: -1.5739933, upper bound: 1.6175560
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 35.12
Output dim: 9, lower bound: -1.5762617, upper bound: 1.6174499
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 35.12
Output dim: 9, lower bound: -1.5765102, upper bound: 1.6175576
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 35.12
Output dim: 9, lower bound: -1.5730927, upper bound: 1.6170486
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 35.12
Output dim: 9, lower bound: -1.5780802, upper bound: 1.6173714
IS_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 35.12
Output dim: 9, lower bound: -1.5805177, upper bound: 1.6149465
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 35.12
Output dim: 9, lower bound: -1.5805175, upper bound: 1.6173776

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -9.7123051, -5.1876278, -9.6209812, -5.2399645, -3.5075283, 3.6179113
1: -17.1165276, -13.3465271, -17.0436935, -13.4041328, -3.3959284, 3.4084225
2: -8.1489286, -4.2811174, -7.9910197, -4.3483791, -3.0673056, 3.1728892
3: -13.8078022, -8.7528582, -13.6998005, -8.8204498, -4.8816147, 4.9399376
4: -3.8904812, -0.2835081, -3.8142135, -0.3522913, -3.2011962, 3.1491280
5: -13.9641304, -9.9900074, -13.9082413, -10.0509338, -2.8220892, 2.8422184
6: -15.9381104, -11.4125967, -15.8663378, -11.4802084, -3.6325626, 3.7099261
7: -8.3667259, -4.2022996, -8.2773657, -4.3769999, -3.8647032, 3.8286643
8: -6.6998405, -2.9925127, -6.5738010, -3.0751076, -3.5630627, 3.5812883
9: 3.9161634, 6.5552163, 4.0191197, 6.4482560, -2.5320926, 2.5360966

Time for backsubstitution: 14.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4608

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5706604, upper bound: 1.6170425
time: 4.75 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5723655, upper bound: 1.6170412
time: 4.98 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -9.7140036, -5.1843190, -9.6318531, -5.2233400, -3.5215607, 3.6389074
1: -17.1194725, -13.3456497, -17.0591755, -13.3986721, -3.4055943, 3.4202652
2: -8.1602211, -4.2808328, -8.0483131, -4.3413038, -3.1068850, 3.1965070
3: -13.8088732, -8.7460575, -13.7091122, -8.7853680, -4.9062185, 4.9597015
4: -3.8913062, -0.2773978, -3.8211176, -0.3208203, -3.2176037, 3.1644602
5: -13.9653378, -9.9875774, -13.9160423, -10.0382614, -2.8339405, 2.8580790
6: -15.9417152, -11.4114227, -15.8850079, -11.4721928, -3.6465855, 3.7212915
7: -8.3680077, -4.1897736, -8.2901201, -4.3139429, -3.8952827, 3.8781610
8: -6.7104511, -2.9915304, -6.6279225, -3.0649648, -3.5992279, 3.6363921
9: 3.9149723, 6.5601497, 4.0106573, 6.4734392, -2.5584669, 2.5494924

Time for backsubstitution: 14.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5820

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5771343, upper bound: 1.6123637
time: 5.44 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5771345, upper bound: 1.6123636
time: 5.08 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.7212048, -5.1725135, -9.6324787, -5.2194204, -3.5344615, 3.6439929
1: -17.1234245, -13.3353519, -17.0599556, -13.3951731, -3.4116955, 3.4337444
2: -8.1695251, -4.2664709, -8.0496616, -4.3364592, -3.1258054, 3.2418683
3: -13.8282280, -8.7362194, -13.7152567, -8.7844362, -4.9272432, 4.9638863
4: -3.9074388, -0.2440495, -3.8215566, -0.3094683, -3.2427602, 3.1637983
5: -13.9940948, -9.9757042, -13.9259071, -10.0381718, -2.8383093, 2.8658738
6: -15.9595442, -11.4041052, -15.8907375, -11.4718847, -3.6536074, 3.7343574
7: -8.3881588, -4.1793408, -8.2958994, -4.3132238, -3.9400692, 3.8891220
8: -6.7207456, -2.9882364, -6.6310463, -3.0642791, -3.6181650, 3.6428099
9: 3.9066763, 6.5659075, 4.0080872, 6.4742870, -2.5676107, 2.5578203

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5820

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5796548, upper bound: 1.6099406
time: 8.20 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5797771, upper bound: 1.6149392
time: 5.72 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.7076492, -5.1849055, -9.6772642, -5.1848912, -3.4987049, 3.6821096
1: -17.1188965, -13.3459063, -17.0944023, -13.3782177, -3.3956614, 3.4496737
2: -8.1580744, -4.2811794, -8.1080341, -4.3008456, -3.1125402, 3.2860167
3: -13.8053732, -8.7464075, -13.7367439, -8.7585011, -4.8987856, 4.9787807
4: -3.8911319, -0.2797561, -3.8348901, -0.3118409, -3.2399459, 3.1545424
5: -13.9618120, -9.9878006, -13.9427547, -10.0155592, -2.8109131, 2.8827877
6: -15.9375191, -11.4119759, -15.9141655, -11.4412899, -3.6230259, 3.7539411
7: -8.3677177, -4.1920586, -8.3161535, -4.2884932, -3.9627590, 3.8913250
8: -6.7058086, -2.9920368, -6.6574125, -3.0332112, -3.5697284, 3.6653757
9: 3.9152546, 6.5571580, 3.9857845, 6.5000086, -2.5847540, 2.5698712

Time for backsubstitution: 14.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5737442, upper bound: 1.5955418
time: 5.20 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5737442, upper bound: 1.6174501
time: 5.41 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.7139730, -5.1843200, -9.7018681, -5.1494989, -3.7004166, 3.7036176
1: -17.1194687, -13.3456783, -17.0972519, -13.3761187, -3.4602528, 3.4499884
2: -8.1601963, -4.2808352, -8.1164789, -4.2884350, -3.3033690, 3.2973547
3: -13.8088646, -8.7460585, -13.7509127, -8.7372808, -5.0141754, 4.9911814
4: -3.8913066, -0.2774167, -3.8496776, -0.3023286, -3.2482562, 3.1968575
5: -13.9653177, -9.9875774, -13.9563580, -9.9956226, -2.8975887, 2.8945169
6: -15.9417019, -11.4114380, -15.9304276, -11.4155197, -3.7792616, 3.7678676
7: -8.3680067, -4.1898022, -8.3325462, -4.2792931, -3.9710131, 3.9655914
8: -6.7104254, -2.9915347, -6.6755128, -3.0053048, -3.7051206, 3.6839781
9: 3.9150076, 6.5601444, 3.9677935, 6.5114498, -2.5964422, 2.5923510

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5820

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5738626, upper bound: 1.6125406
time: 5.22 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5739876, upper bound: 1.6175514
time: 6.01 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -9.7148504, -5.1731048, -9.6778927, -5.1809711, -3.5098171, 3.6875136
1: -17.1228523, -13.3356094, -17.0951767, -13.3747139, -3.4023495, 3.4576955
2: -8.1673803, -4.2668219, -8.1093798, -4.2960014, -3.1259537, 3.3000484
3: -13.8247252, -8.7365685, -13.7428923, -8.7575665, -4.9062386, 4.9945459
4: -3.9072609, -0.2464088, -3.8353286, -0.3004856, -3.2621841, 3.1538715
5: -13.9905643, -9.9759274, -13.9526024, -10.0154705, -2.8132315, 2.9009023
6: -15.9553518, -11.4046535, -15.9198980, -11.4409752, -3.6296730, 3.7673750
7: -8.3878717, -4.1816235, -8.3218966, -4.2877760, -3.9717512, 3.9043565
8: -6.7161026, -2.9887409, -6.6605368, -3.0325303, -3.5882607, 3.6717958
9: 3.9069567, 6.5629168, 3.9832144, 6.5008540, -2.5938973, 2.5797024

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5820
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 494

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5762617, upper bound: 1.5955414
time: 5.35 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5762617, upper bound: 1.6174490
time: 4.96 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 25.00 seconds
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 25.00
Output dim: 9, lower bound: -1.5706604, upper bound: 1.6170425
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 25.00
Output dim: 9, lower bound: -1.5723655, upper bound: 1.6170412
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 25.00
Output dim: 9, lower bound: -1.5771343, upper bound: 1.6123637
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 25.00
Output dim: 9, lower bound: -1.5771345, upper bound: 1.6123636
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 25.00
Output dim: 9, lower bound: -1.5796548, upper bound: 1.6099406
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 25.00
Output dim: 9, lower bound: -1.5797771, upper bound: 1.6149392
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 25.00
Output dim: 9, lower bound: -1.5737442, upper bound: 1.5955418
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 25.00
Output dim: 9, lower bound: -1.5737442, upper bound: 1.6174501
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 25.00
Output dim: 9, lower bound: -1.5738626, upper bound: 1.6125406
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 25.00
Output dim: 9, lower bound: -1.5739876, upper bound: 1.6175514
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 25.00
Output dim: 9, lower bound: -1.5762617, upper bound: 1.5955414
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 25.00
Output dim: 9, lower bound: -1.5762617, upper bound: 1.6174490
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 25.00
Output dim: 9, lower bound: -1.5765102, upper bound: 1.6175576
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 25.00
Output dim: 9, lower bound: -1.5730927, upper bound: 1.6170486
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 25.00
Output dim: 9, lower bound: -1.5780802, upper bound: 1.6173714
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 25.00
Output dim: 9, lower bound: -1.5805175, upper bound: 1.6173776
Binary search (step 2): status=Status.UNKNOWN, k_low=6, k_high=6, k_mid=6, eps_mid=0.0234375, abs_max=2.6592936515808105
rel_dist={9: [-1.6176245637311917, 1.6176249981850015]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01953125
execution time: 2420.26 seconds
