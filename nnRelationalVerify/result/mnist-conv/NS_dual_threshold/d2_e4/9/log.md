## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 9)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.6361788214999999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.9221611, -5.9657760, -8.9221611, -5.9657760, -1.7538667, 1.7538671)
1: (-16.5001736, -14.1283016, -16.5001736, -14.1283016, -1.5473294, 1.5473294)
2: (-7.4758034, -4.9801350, -7.4758034, -4.9801350, -1.5473089, 1.5473089)
3: (-12.9350147, -9.6876116, -12.9350147, -9.6876116, -2.6015015, 2.6015015)
4: (-3.2633729, -0.9272225, -3.2633729, -0.9272225, -1.7983255, 1.7983255)
5: (-13.3927860, -10.7282457, -13.3927860, -10.7282457, -1.2112970, 1.2112970)
6: (-15.2848930, -12.3271980, -15.2848930, -12.3271980, -1.7188849, 1.7188859)
7: (-7.6543331, -5.0435581, -7.6543331, -5.0435581, -2.1962242, 2.1962242)
8: (-6.0455451, -3.6904855, -6.0455451, -3.6904855, -1.5854363, 1.5854363)
9: (4.5871305, 6.2297759, 4.5871305, 6.2297759, -1.4497461, 1.4497461)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.52 + 35.04 = 57.56 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.6393757, upper bound: 0.6393755

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5875
type: B, layer: 1, pos: 5875
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 4557
type: B, layer: 1, pos: 4557
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 5875

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6393756, upper bound: 0.6332733
time: 4.14 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6393748, upper bound: 0.6393744
time: 4.30 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 8.69 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 8.69
Output dim: 9, lower bound: -0.6393756, upper bound: 0.6332733
NS_A2, status: Status.UNKNOWN, split count: 1, time: 8.69
Output dim: 9, lower bound: -0.6393748, upper bound: 0.6393744

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -8.8729248, -5.9873099, -8.8933964, -5.9659896, -1.7045584, 1.7033424
1: -16.4740753, -14.1397972, -16.4852581, -14.1285038, -1.5202270, 1.5204244
2: -7.4266262, -5.0024662, -7.4473104, -4.9802017, -1.4980989, 1.4964061
3: -12.9051704, -9.7031927, -12.9179564, -9.6885529, -2.5708361, 2.5690641
4: -3.2553582, -0.9402652, -3.2626915, -0.9345481, -1.7810369, 1.7844715
5: -13.3643789, -10.7405643, -13.3761597, -10.7282848, -1.1828656, 1.1822691
6: -15.2521486, -12.3448048, -15.2657490, -12.3283195, -1.6848140, 1.6818156
7: -7.6436257, -5.0673780, -7.6541519, -5.0573850, -2.1714525, 2.1715841
8: -6.0116177, -3.7082419, -6.0258384, -3.6912508, -1.5504441, 1.5470572
9: 4.6009645, 6.2021103, 4.5875540, 6.2136059, -1.4197073, 1.4216323

Time for backsubstitution: 21.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 4557
type: A, layer: 1, pos: 4557
type: B, layer: 1, pos: 5875
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of NS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 961

### Candidate
type: B, layer: 1, pos: 4557

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.6301898, upper bound: 0.6315914
time: 4.22 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6393553, upper bound: 0.6332540
time: 4.54 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -8.9221592, -5.9657764, -8.9221611, -5.9657769, -1.6962695, 1.7458458
1: -16.5001678, -14.1283035, -16.5001698, -14.1283035, -1.5244665, 1.5473266
2: -7.4757953, -4.9801364, -7.4757981, -4.9801350, -1.4919467, 1.5463099
3: -12.9350128, -9.6876135, -12.9350166, -9.6876144, -2.5656881, 2.6014977
4: -3.2633717, -0.9272215, -3.2633715, -0.9272237, -1.7977104, 1.7810879
5: -13.3927841, -10.7282457, -13.3927860, -10.7282467, -1.1775236, 1.2112617
6: -15.2848911, -12.3271942, -15.2848930, -12.3271980, -1.6786795, 1.7188830
7: -7.6543331, -5.0435629, -7.6543331, -5.0435624, -2.1962233, 2.1747274
8: -6.0455408, -3.6904860, -6.0455465, -3.6904879, -1.5413556, 1.5854220
9: 4.5871310, 6.2297759, 4.5871315, 6.2297754, -1.4497466, 1.4150481

Time for backsubstitution: 21.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 4557
type: A, layer: 1, pos: 4557
type: B, layer: 1, pos: 5875
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906

Time for candidate selection: 0.32 seconds

### Candidate
type: B, layer: 1, pos: 961

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6319945, upper bound: 0.6392640
time: 4.09 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6392645, upper bound: 0.6392639
time: 3.98 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 29.71 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 29.71
Output dim: 9, lower bound: -0.6301898, upper bound: 0.6315914
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 29.71
Output dim: 9, lower bound: -0.6393553, upper bound: 0.6332540
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 29.71
Output dim: 9, lower bound: -0.6319945, upper bound: 0.6392640
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 29.71
Output dim: 9, lower bound: -0.6392645, upper bound: 0.6392639

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -8.8729248, -5.9873099, -8.8933945, -5.9659891, -1.7043672, 1.7022882
1: -16.4740753, -14.1397972, -16.4852562, -14.1285057, -1.5202246, 1.5194435
2: -7.4266262, -5.0024662, -7.4473000, -4.9802032, -1.4980960, 1.4826779
3: -12.9051704, -9.7031927, -12.9179459, -9.6885567, -2.5708351, 2.5606003
4: -3.2553582, -0.9402652, -3.2626805, -0.9345515, -1.7795405, 1.7570534
5: -13.3643789, -10.7405643, -13.3761597, -10.7282848, -1.1630554, 1.1822672
6: -15.2521486, -12.3448048, -15.2657480, -12.3283253, -1.6848106, 1.6844010
7: -7.6436257, -5.0673780, -7.6541471, -5.0573955, -2.1444283, 2.1684837
8: -6.0116177, -3.7082419, -6.0258360, -3.6912565, -1.5263724, 1.5470529
9: 4.6009645, 6.2021103, 4.5875583, 6.2135963, -1.4033265, 1.4216280

Time for backsubstitution: 21.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 5875
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4557

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 961

### Candidate
type: B, layer: 1, pos: 961

### Candidate
type: B, layer: 1, pos: 5875

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.6341690, upper bound: 0.6332539
time: 4.18 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.6341690, upper bound: 0.6332557
time: 4.13 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -8.9146938, -5.9657998, -8.9082870, -5.9658217, -1.6887646, 1.7319427
1: -16.4996758, -14.1283302, -16.4992752, -14.1283550, -1.5236683, 1.5459352
2: -7.4733310, -4.9801731, -7.4712181, -4.9801979, -1.4894018, 1.5416765
3: -12.9308519, -9.6877794, -12.9272804, -9.6879177, -2.5612659, 2.5936184
4: -3.2632658, -0.9300327, -3.2631779, -0.9324450, -1.7922468, 1.7787657
5: -13.3886423, -10.7282543, -13.3850861, -10.7282543, -1.1733274, 1.2035222
6: -15.2799587, -12.3273754, -15.2757244, -12.3275270, -1.6733460, 1.7094870
7: -7.6542921, -5.0461841, -7.6542597, -5.0484338, -2.1911945, 2.1719761
8: -6.0401363, -3.6906071, -6.0354958, -3.6907034, -1.5356393, 1.5751266
9: 4.5871849, 6.2262392, 4.5872135, 6.2232032, -1.4431324, 1.4114428

Time for backsubstitution: 21.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4557
type: A, layer: 1, pos: 4557
type: B, layer: 1, pos: 5875
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 4557

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6228040, upper bound: 0.6375733
time: 4.50 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6319740, upper bound: 0.6392447
time: 4.28 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -8.9221230, -5.9657764, -8.9222383, -5.9411755, -1.7096577, 1.7375677
1: -16.5001678, -14.1283321, -16.5006275, -14.1273117, -1.5257921, 1.5469899
2: -7.4757700, -4.9801350, -7.4759760, -4.9718385, -1.5002103, 1.5439377
3: -12.9350033, -9.6876144, -12.9355164, -9.6730547, -2.5802126, 2.5971985
4: -3.2633719, -0.9272482, -3.2736058, -0.9268928, -1.7944646, 1.7895484
5: -13.3927622, -10.7282486, -13.3927727, -10.7145796, -1.1898737, 1.2064784
6: -15.2848740, -12.3272095, -15.2849922, -12.3099022, -1.6959367, 1.7133036
7: -7.6543331, -5.0435967, -7.6631866, -5.0435390, -2.1937838, 2.1834307
8: -6.0455151, -3.6904874, -6.0456862, -3.6714702, -1.5602217, 1.5790443
9: 4.5871716, 6.2297697, 4.5746913, 6.2298183, -1.4456449, 1.4274483

Time for backsubstitution: 22.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4557
type: A, layer: 1, pos: 4557
type: B, layer: 1, pos: 5875
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 4557

## Relational analysis of NS_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6300728, upper bound: 0.6375732
time: 4.47 seconds

## Relational analysis of NS_A2_B2_B2

### Relational analysis result of NS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6392441, upper bound: 0.6392447
time: 4.17 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 30.96 seconds
NS_A1_B2_B1, status: Status.VERIFIED, split count: 3, time: 30.96
Output dim: 9, lower bound: -0.6341690, upper bound: 0.6332539
NS_A1_B2_B2, status: Status.VERIFIED, split count: 3, time: 30.96
Output dim: 9, lower bound: -0.6341690, upper bound: 0.6332557
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 30.96
Output dim: 9, lower bound: -0.6228040, upper bound: 0.6375733
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 30.96
Output dim: 9, lower bound: -0.6319740, upper bound: 0.6392447
NS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 30.96
Output dim: 9, lower bound: -0.6300728, upper bound: 0.6375732
NS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 30.96
Output dim: 9, lower bound: -0.6392441, upper bound: 0.6392447

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: -8.9103765, -5.9668522, -8.8972034, -5.9690156, -1.6802673, 1.7188487
1: -16.4928398, -14.1350679, -16.4842682, -14.1488314, -1.4976540, 1.5231442
2: -7.4600601, -4.9855504, -7.4383640, -4.9971271, -1.4569330, 1.5016546
3: -12.9131584, -9.6914845, -12.8845100, -9.6995869, -2.5341339, 2.5481033
4: -3.2399523, -0.9363956, -3.2104390, -0.9545116, -1.7434244, 1.7245283
5: -13.3849831, -10.7393608, -13.3717966, -10.7548800, -1.1440310, 1.1697364
6: -15.2747173, -12.3356228, -15.2615261, -12.3479128, -1.6466284, 1.6840138
7: -7.6457424, -5.0763874, -7.6280017, -5.1196332, -2.1172829, 2.1089821
8: -6.0350466, -3.7031565, -6.0171576, -3.7209306, -1.4962006, 1.5272298
9: 4.5994129, 6.2091579, 4.6267023, 6.1819162, -1.3923326, 1.3610926

Time for backsubstitution: 21.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5875
type: A, layer: 1, pos: 4557
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 5875

## Relational analysis of NS_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4557

## Relational analysis of NS_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.6228040, upper bound: 0.6300721
time: 4.34 seconds

## Relational analysis of NS_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6228040, upper bound: 0.6375734
time: 4.20 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: -8.9146938, -5.9657998, -8.9082823, -5.9658241, -1.6885719, 1.7308156
1: -16.4996758, -14.1283302, -16.4992695, -14.1283588, -1.5236650, 1.5449562
2: -7.4733310, -4.9801731, -7.4712090, -4.9801979, -1.4893994, 1.5279226
3: -12.9308519, -9.6877794, -12.9272690, -9.6879215, -2.5612650, 2.5851574
4: -3.2632658, -0.9300327, -3.2631679, -0.9324467, -1.7907548, 1.7513509
5: -13.3886423, -10.7282543, -13.3850851, -10.7282600, -1.1535149, 1.2001836
6: -15.2799587, -12.3273754, -15.2757206, -12.3275299, -1.6733427, 1.7120714
7: -7.6542921, -5.0461841, -7.6542583, -5.0484438, -2.1642218, 2.1688757
8: -6.0401363, -3.6906071, -6.0354939, -3.6907082, -1.5115685, 1.5740499
9: 4.5871849, 6.2262392, 4.5872183, 6.2231922, -1.4267507, 1.4114385

Time for backsubstitution: 21.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5875
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4557

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 5875

## Relational analysis of NS_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of NS_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6319740, upper bound: 0.6369265
time: 4.19 seconds

## Relational analysis of NS_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6319740, upper bound: 0.6392447
time: 4.16 seconds

## BFS NS instance: NS_A2_B2_B1

### Backsubstitution after applying NS history:
0: -8.9178076, -5.9668245, -8.9111519, -5.9443607, -1.7006335, 1.7244744
1: -16.4933281, -14.1350689, -16.4856033, -14.1477814, -1.4997778, 1.5241804
2: -7.4624958, -4.9855127, -7.4431229, -4.9887495, -1.4677587, 1.5039136
3: -12.9173098, -9.6913185, -12.8927526, -9.6847162, -2.5530853, 2.5516796
4: -3.2400568, -0.9336121, -3.2208707, -0.9489336, -1.7456598, 1.7353654
5: -13.3890991, -10.7393541, -13.3794823, -10.7411976, -1.1605175, 1.1727779
6: -15.2796421, -12.3354578, -15.2707844, -12.3302994, -1.6692080, 1.6878176
7: -7.6457801, -5.0738001, -7.6368070, -5.1147413, -2.1198635, 2.1148634
8: -6.0404196, -3.7030382, -6.0273294, -3.7016869, -1.5207992, 1.5311768
9: 4.5994053, 6.2126889, 4.6141500, 6.1885295, -1.3948402, 1.3692551

Time for backsubstitution: 22.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5875
type: A, layer: 1, pos: 4557
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 5875

## Relational analysis of NS_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4557

## Relational analysis of NS_A2_B2_B1_A1

### Relational analysis result of NS_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.6300726, upper bound: 0.6300721
time: 4.38 seconds

## Relational analysis of NS_A2_B2_B1_A2

### Relational analysis result of NS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6300726, upper bound: 0.6375732
time: 4.26 seconds

## BFS NS instance: NS_A2_B2_B2

### Backsubstitution after applying NS history:
0: -8.9221230, -5.9657764, -8.9222355, -5.9411745, -1.7091894, 1.7364450
1: -16.5001678, -14.1283321, -16.5006218, -14.1273155, -1.5257893, 1.5460205
2: -7.4757700, -4.9801350, -7.4759703, -4.9718390, -1.5002089, 1.5301843
3: -12.9350033, -9.6876144, -12.9355078, -9.6730566, -2.5802116, 2.5887365
4: -3.2633719, -0.9272482, -3.2735953, -0.9268963, -1.7929688, 1.7621298
5: -13.3927622, -10.7282486, -13.3927698, -10.7145844, -1.1700490, 1.2032282
6: -15.2848740, -12.3272095, -15.2849932, -12.3099070, -1.6959324, 1.7158885
7: -7.6543331, -5.0435967, -7.6631832, -5.0435500, -2.1668129, 2.1803265
8: -6.0455151, -3.6904874, -6.0456824, -3.6714773, -1.5361514, 1.5779982
9: 4.5871716, 6.2297697, 4.5746946, 6.2298079, -1.4292636, 1.4274454

Time for backsubstitution: 22.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5875
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 4608
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4557

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 5875

## Relational analysis of NS_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of NS_A2_B2_B2_A1

### Relational analysis result of NS_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.6319740, upper bound: 0.6319755
time: 4.21 seconds

## Relational analysis of NS_A2_B2_B2_A2

### Relational analysis result of NS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6319741, upper bound: 0.6392445
time: 4.08 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 36.56 seconds
NS_A2_B1_B1_A1, status: Status.VERIFIED, split count: 4, time: 36.56
Output dim: 9, lower bound: -0.6228040, upper bound: 0.6300721
NS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 36.56
Output dim: 9, lower bound: -0.6228040, upper bound: 0.6375734
NS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 36.56
Output dim: 9, lower bound: -0.6319740, upper bound: 0.6369265
NS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 36.56
Output dim: 9, lower bound: -0.6319740, upper bound: 0.6392447
NS_A2_B2_B1_A1, status: Status.VERIFIED, split count: 4, time: 36.56
Output dim: 9, lower bound: -0.6300726, upper bound: 0.6300721
NS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 36.56
Output dim: 9, lower bound: -0.6300726, upper bound: 0.6375732
NS_A2_B2_B2_A1, status: Status.VERIFIED, split count: 4, time: 36.56
Output dim: 9, lower bound: -0.6319740, upper bound: 0.6319755
NS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 36.56
Output dim: 9, lower bound: -0.6319741, upper bound: 0.6392445

## BFS NS instance: NS_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -8.9146500, -5.9662237, -8.8972034, -5.9690156, -1.6852655, 1.7190351
1: -16.4967098, -14.1285172, -16.4842682, -14.1488314, -1.5023088, 1.5285611
2: -7.4732504, -4.9814382, -7.4383640, -4.9971271, -1.4688911, 1.5030279
3: -12.9307995, -9.6886625, -12.8845100, -9.6995869, -2.5518522, 2.5502558
4: -3.2569613, -0.9301836, -3.2104390, -0.9545116, -1.7473550, 1.7297077
5: -13.3884296, -10.7284813, -13.3717966, -10.7548800, -1.1473713, 1.1708341
6: -15.2783871, -12.3274450, -15.2615261, -12.3479128, -1.6509018, 1.6928759
7: -7.6539669, -5.0507884, -7.6280017, -5.1196332, -2.1213188, 2.1145487
8: -6.0399203, -3.6909065, -6.0171576, -3.7209306, -1.5043173, 1.5290709
9: 4.5887542, 6.2262268, 4.6267023, 6.1819162, -1.4002638, 1.3663585

Time for backsubstitution: 22.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5875
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 5875

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of NS_A2_B1_B1_A2_A1

### Relational analysis result of NS_A2_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.6228040, upper bound: 0.6352556
time: 4.54 seconds

## Relational analysis of NS_A2_B1_B1_A2_A2

### Relational analysis result of NS_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6228040, upper bound: 0.6375733
time: 4.35 seconds

## BFS NS instance: NS_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -8.9082870, -5.9658222, -8.9082823, -5.9658241, -1.6828375, 1.7307928
1: -16.4992714, -14.1283550, -16.4992695, -14.1283588, -1.5230474, 1.5449319
2: -7.4712152, -4.9801970, -7.4712090, -4.9801979, -1.4872861, 1.5278873
3: -12.9272757, -9.6879196, -12.9272690, -9.6879215, -2.5576897, 2.5850382
4: -3.2631772, -0.9324455, -3.2631679, -0.9324467, -1.7906508, 1.7513280
5: -13.3850880, -10.7282562, -13.3850851, -10.7282600, -1.1500673, 1.2001712
6: -15.2757206, -12.3275280, -15.2757206, -12.3275299, -1.6690998, 1.7118959
7: -7.6542602, -5.0484352, -7.6542583, -5.0484438, -2.1641932, 2.1665716
8: -6.0354939, -3.6907029, -6.0354939, -3.6907082, -1.5087280, 1.5739405
9: 4.5872126, 6.2232018, 4.5872183, 6.2231922, -1.4267244, 1.4083996

Time for backsubstitution: 22.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5875
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4557

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 5875

### Candidate
type: A, layer: 1, pos: 5798

## Relational analysis of NS_A2_B1_B2_A1_A1

### Relational analysis result of NS_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6308862, upper bound: 0.6368330
time: 4.33 seconds

## Relational analysis of NS_A2_B1_B2_A1_A2

### Relational analysis result of NS_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6319731, upper bound: 0.6369257
time: 4.37 seconds

## BFS NS instance: NS_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -8.9222336, -5.9411783, -8.9082823, -5.9658241, -1.6961040, 1.7308934
1: -16.5006180, -14.1273117, -16.4992695, -14.1283588, -1.5243831, 1.5459232
2: -7.4759741, -4.9718428, -7.4712090, -4.9801979, -1.4920015, 1.5280809
3: -12.9355154, -9.6730576, -12.9272690, -9.6879215, -2.5659065, 2.5998383
4: -3.2736056, -0.9268937, -3.2631679, -0.9324467, -1.8009925, 1.7536054
5: -13.3927727, -10.7145824, -13.3850851, -10.7282600, -1.1576414, 1.2002046
6: -15.2849884, -12.3099041, -15.2757206, -12.3275299, -1.6783600, 1.7228484
7: -7.6631832, -5.0435414, -7.6542583, -5.0484438, -2.1720352, 2.1715536
8: -6.0456834, -3.6714725, -6.0354939, -3.6907082, -1.5170403, 1.5742993
9: 4.5746965, 6.2298183, 4.5872183, 6.2231922, -1.4355221, 1.4150157

Time for backsubstitution: 22.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5875
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4557

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 5875

### Candidate
type: A, layer: 1, pos: 5798

## Relational analysis of NS_A2_B1_B2_A2_A1

### Relational analysis result of NS_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6308862, upper bound: 0.6391512
time: 4.01 seconds

## Relational analysis of NS_A2_B1_B2_A2_A2

### Relational analysis result of NS_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6319731, upper bound: 0.6392439
time: 4.24 seconds

## BFS NS instance: NS_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -8.9220800, -5.9661980, -8.9111519, -5.9443607, -1.7039266, 1.7246609
1: -16.4972038, -14.1285162, -16.4856033, -14.1477814, -1.5044351, 1.5295973
2: -7.4756908, -4.9813995, -7.4431229, -4.9887495, -1.4719768, 1.5052867
3: -12.9349537, -9.6884966, -12.8927526, -9.6847162, -2.5708036, 2.5538330
4: -3.2570643, -0.9274008, -3.2208707, -0.9489336, -1.7496963, 1.7405038
5: -13.3925486, -10.7284756, -13.3794823, -10.7411976, -1.1604698, 1.1738760
6: -15.2833080, -12.3272810, -15.2707844, -12.3302994, -1.6734824, 1.6966801
7: -7.6540089, -5.0482035, -7.6368070, -5.1147413, -2.1239033, 2.1184177
8: -6.0452981, -3.6907864, -6.0273294, -3.7016869, -1.5261788, 1.5330176
9: 4.5887423, 6.2297564, 4.6141500, 6.1885295, -1.4027729, 1.3707891

Time for backsubstitution: 22.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5875
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 5875

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of NS_A2_B2_B1_A2_A1

### Relational analysis result of NS_A2_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.6228040, upper bound: 0.6303062
time: 4.37 seconds

## Relational analysis of NS_A2_B2_B1_A2_A2

### Relational analysis result of NS_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6228042, upper bound: 0.6375732
time: 4.46 seconds

## BFS NS instance: NS_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -8.9222336, -5.9411783, -8.9222355, -5.9411745, -1.6886649, 1.7376809
1: -16.5006180, -14.1273117, -16.5006218, -14.1273155, -1.5260248, 1.5481062
2: -7.4759741, -4.9718428, -7.4759703, -4.9718390, -1.4905591, 1.5321875
3: -12.9355154, -9.6730576, -12.9355078, -9.6730566, -2.5620213, 2.5893679
4: -3.2736056, -0.9268937, -3.2735953, -0.9268963, -1.7932129, 1.7541566
5: -13.3927727, -10.7145824, -13.3927698, -10.7145844, -1.1531754, 1.2042186
6: -15.2849884, -12.3099041, -15.2849932, -12.3099070, -1.6740036, 1.7168016
7: -7.6631832, -5.0435414, -7.6631832, -5.0435500, -2.1685038, 2.1708508
8: -6.0456834, -3.6714725, -6.0456824, -3.6714773, -1.5131636, 1.5794373
9: 4.5746965, 6.2298183, 4.5746946, 6.2298079, -1.4297738, 1.4114571

Time for backsubstitution: 22.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5875
type: A, layer: 1, pos: 5798
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4557

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 5875

### Candidate
type: A, layer: 1, pos: 5798

## Relational analysis of NS_A2_B2_B2_A2_A1

### Relational analysis result of NS_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6308862, upper bound: 0.6391511
time: 4.20 seconds

## Relational analysis of NS_A2_B2_B2_A2_A2

### Relational analysis result of NS_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6319731, upper bound: 0.6392438
time: 4.54 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 31.28 seconds
NS_A2_B1_B1_A2_A1, status: Status.VERIFIED, split count: 5, time: 31.28
Output dim: 9, lower bound: -0.6228040, upper bound: 0.6352556
NS_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 31.28
Output dim: 9, lower bound: -0.6228040, upper bound: 0.6375733
NS_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 31.28
Output dim: 9, lower bound: -0.6308862, upper bound: 0.6368330
NS_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 31.28
Output dim: 9, lower bound: -0.6319731, upper bound: 0.6369257
NS_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 31.28
Output dim: 9, lower bound: -0.6308862, upper bound: 0.6391512
NS_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 31.28
Output dim: 9, lower bound: -0.6319731, upper bound: 0.6392439
NS_A2_B2_B1_A2_A1, status: Status.VERIFIED, split count: 5, time: 31.28
Output dim: 9, lower bound: -0.6228040, upper bound: 0.6303062
NS_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 31.28
Output dim: 9, lower bound: -0.6228042, upper bound: 0.6375732
NS_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 31.28
Output dim: 9, lower bound: -0.6308862, upper bound: 0.6391511
NS_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 31.28
Output dim: 9, lower bound: -0.6319731, upper bound: 0.6392438

## BFS NS instance: NS_A2_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -8.9221916, -5.9415979, -8.8972034, -5.9690156, -1.6927986, 1.7191133
1: -16.4976654, -14.1274948, -16.4842682, -14.1488314, -1.5030327, 1.5295267
2: -7.4758930, -4.9731069, -7.4383640, -4.9971271, -1.4692297, 1.5031862
3: -12.9354649, -9.6739368, -12.8845100, -9.6995869, -2.5564938, 2.5649405
4: -3.2673006, -0.9270470, -3.2104390, -0.9545116, -1.7476721, 1.7319403
5: -13.3925571, -10.7148094, -13.3717966, -10.7548800, -1.1514974, 1.1708555
6: -15.2834206, -12.3099766, -15.2615261, -12.3479128, -1.6559191, 1.6950779
7: -7.6628613, -5.0481472, -7.6280017, -5.1196332, -2.1253252, 2.1149669
8: -6.0454674, -3.6717710, -6.0171576, -3.7209306, -1.5097890, 1.5293212
9: 4.5762634, 6.2298045, 4.6267023, 6.1819162, -1.4026923, 1.3664465

Time for backsubstitution: 22.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5875
type: B, layer: 1, pos: 5798
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 5875

### Candidate
type: B, layer: 1, pos: 5798

## Relational analysis of NS_A2_B1_B1_A2_A2_B1

### Relational analysis result of NS_A2_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6227131, upper bound: 0.6364887
time: 4.36 seconds

## Relational analysis of NS_A2_B1_B1_A2_A2_B2

### Relational analysis result of NS_A2_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6228032, upper bound: 0.6375706
time: 4.57 seconds

## BFS NS instance: NS_A2_B1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -8.9046021, -5.9746313, -8.9075937, -5.9702616, -1.6747818, 1.7212968
1: -16.4970818, -14.1335735, -16.4984169, -14.1310463, -1.5181952, 1.5389500
2: -7.4674950, -4.9903736, -7.4705305, -4.9854422, -1.4779978, 1.5164456
3: -12.9138184, -9.6920958, -12.9205074, -9.6886148, -2.5431690, 2.5720739
4: -3.2563639, -0.9574170, -3.2629228, -0.9452519, -1.7683868, 1.7265501
5: -13.3635120, -10.7331152, -13.3740368, -10.7283115, -1.1285286, 1.1779957
6: -15.2633057, -12.3302441, -15.2694683, -12.3276424, -1.6566668, 1.7000446
7: -7.6406751, -5.0524974, -7.6475215, -5.0487871, -2.1502113, 2.1556578
8: -6.0317535, -3.6928711, -6.0336647, -3.6915822, -1.5022140, 1.5693786
9: 4.5931864, 6.2203145, 4.5901575, 6.2224722, -1.4203382, 1.4027190

Time for backsubstitution: 22.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5875
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 4557

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 5875

### Candidate
type: A, layer: 1, pos: 4608

## Relational analysis of NS_A2_B1_B2_A1_A1_A1

### Relational analysis result of NS_A2_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6352733, upper bound: 0.6368967
time: 4.06 seconds

## Relational analysis of NS_A2_B1_B2_A1_A1_A2

### Relational analysis result of NS_A2_B1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6359012, upper bound: 0.6368967
time: 4.51 seconds

## BFS NS instance: NS_A2_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -8.9082851, -5.9658232, -8.9082832, -5.9658232, -1.6828375, 1.7249949
1: -16.4992714, -14.1283569, -16.4992676, -14.1283579, -1.5237508, 1.5448823
2: -7.4712152, -4.9801989, -7.4712110, -4.9802012, -1.4872866, 1.5213299
3: -12.9272785, -9.6879225, -12.9272709, -9.6879225, -2.5506411, 2.5839605
4: -3.2631757, -0.9324477, -3.2631679, -0.9324474, -1.7894192, 1.7267709
5: -13.3850822, -10.7282562, -13.3850861, -10.7282600, -1.1293206, 1.1954167
6: -15.2757215, -12.3275290, -15.2757215, -12.3275299, -1.6638594, 1.7104635
7: -7.6542578, -5.0484352, -7.6542563, -5.0484447, -2.1562452, 2.1665697
8: -6.0354943, -3.6907020, -6.0354939, -3.6907091, -1.5122113, 1.5735683
9: 4.5872135, 6.2232018, 4.5872207, 6.2231932, -1.4253807, 1.4083991

Time for backsubstitution: 22.26 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 57.56 + 551.56 = 609.11 seconds
