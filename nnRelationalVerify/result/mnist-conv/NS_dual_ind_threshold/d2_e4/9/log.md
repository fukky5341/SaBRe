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
execution time: IAR + RelationalAnalysis = 22.92 + 34.83 = 57.75 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.6393757, upper bound: 0.6393755

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5875
type: A, layer: 1, pos: 4557
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 5875

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6393756, upper bound: 0.6332733
time: 4.27 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6393748, upper bound: 0.6393744
time: 4.42 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 8.94 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 8.94
Output dim: 9, lower bound: -0.6393756, upper bound: 0.6332733
NS_A2, status: Status.UNKNOWN, split count: 1, time: 8.94
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

Time for backsubstitution: 20.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4557
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 5875
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 906

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 4557

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.6301898, upper bound: 0.6315914
time: 4.04 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6393553, upper bound: 0.6332540
time: 4.14 seconds

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

Time for backsubstitution: 21.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4557
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 5875
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 906

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 4557

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6301889, upper bound: 0.6376898
time: 4.30 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6393544, upper bound: 0.6393551
time: 4.43 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 30.66 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 30.66
Output dim: 9, lower bound: -0.6301898, upper bound: 0.6315914
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 30.66
Output dim: 9, lower bound: -0.6393553, upper bound: 0.6332540
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 30.66
Output dim: 9, lower bound: -0.6301889, upper bound: 0.6376898
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 30.66
Output dim: 9, lower bound: -0.6393544, upper bound: 0.6393551

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

Time for backsubstitution: 21.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4557
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 4557

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6376910, upper bound: 0.6240899
time: 4.39 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6376912, upper bound: 0.6240898
time: 4.63 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -8.9178429, -5.9668279, -8.9110775, -5.9689679, -1.6877789, 1.7327526
1: -16.4933281, -14.1350422, -16.4851532, -14.1487808, -1.4984455, 1.5245209
2: -7.4625225, -4.9855132, -7.4429407, -4.9970646, -1.4594798, 1.5062847
3: -12.9173183, -9.6913176, -12.8922529, -9.6992779, -2.5385571, 2.5559893
4: -3.2400570, -0.9335847, -3.2106328, -0.9492860, -1.7488813, 1.7268867
5: -13.3891230, -10.7393522, -13.3794985, -10.7548695, -1.1482258, 1.1775022
6: -15.2796564, -12.3354464, -15.2706890, -12.3475838, -1.6519556, 1.6934037
7: -7.6457820, -5.0737667, -7.6280603, -5.1147604, -2.1223059, 2.1117115
8: -6.0404491, -3.7030358, -6.0271997, -3.7207136, -1.5019178, 1.5375631
9: 4.5993595, 6.2126923, 4.6266246, 6.1884871, -1.3989444, 1.3646932

Time for backsubstitution: 21.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 4557
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.6300729, upper bound: 0.6303046
time: 4.32 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6300731, upper bound: 0.6375730
time: 4.55 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -8.9221592, -5.9657764, -8.9221582, -5.9657764, -1.6960773, 1.7447231
1: -16.5001678, -14.1283035, -16.5001678, -14.1283054, -1.5244632, 1.5463476
2: -7.4757953, -4.9801364, -7.4757910, -4.9801373, -1.4919438, 1.5325572
3: -12.9350128, -9.6876135, -12.9350061, -9.6876154, -2.5656872, 2.5930367
4: -3.2633717, -0.9272215, -3.2633617, -0.9272258, -1.7962179, 1.7536697
5: -13.3927841, -10.7282457, -13.3927879, -10.7282486, -1.1577129, 1.2079501
6: -15.2848911, -12.3271942, -15.2848883, -12.3272009, -1.6786766, 1.7214680
7: -7.6543331, -5.0435629, -7.6543293, -5.0435719, -2.1692505, 2.1716261
8: -6.0455408, -3.6904860, -6.0455437, -3.6904945, -1.5172839, 1.5843754
9: 4.5871310, 6.2297759, 4.5871363, 6.2297673, -1.4333658, 1.4150443

Time for backsubstitution: 21.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4557
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 4557

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6376902, upper bound: 0.6301884
time: 4.37 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6376903, upper bound: 0.6393568
time: 4.34 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 30.69 seconds
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 30.69
Output dim: 9, lower bound: -0.6376910, upper bound: 0.6240899
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 30.69
Output dim: 9, lower bound: -0.6376912, upper bound: 0.6240898
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 30.69
Output dim: 9, lower bound: -0.6300729, upper bound: 0.6303046
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 30.69
Output dim: 9, lower bound: -0.6300731, upper bound: 0.6375730
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 30.69
Output dim: 9, lower bound: -0.6376902, upper bound: 0.6301884
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 30.69
Output dim: 9, lower bound: -0.6376903, upper bound: 0.6393568

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -8.8618269, -5.9905033, -8.8933516, -5.9664116, -1.6920061, 1.6998453
1: -16.4590530, -14.1602325, -16.4822941, -14.1286869, -1.5028248, 1.4990988
2: -7.3937712, -5.0193787, -7.4472284, -4.9814672, -1.4629064, 1.4669304
3: -12.8623619, -9.7148638, -12.9179039, -9.6894388, -2.5274477, 2.5596504
4: -3.2026167, -0.9623904, -3.2563815, -0.9347026, -1.7319899, 1.7456784
5: -13.3510914, -10.7671843, -13.3759499, -10.7285109, -1.1624589, 1.1563077
6: -15.2379475, -12.3651743, -15.2641830, -12.3283939, -1.6682043, 1.6593862
7: -7.6175718, -5.1386256, -7.6538291, -5.0619907, -2.1110010, 2.1016827
8: -5.9932923, -3.7384729, -6.0256238, -3.6915498, -1.5203810, 1.5157351
9: 4.6403780, 6.1608276, 4.5891209, 6.2135954, -1.3691940, 1.3787665

Time for backsubstitution: 21.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 5875
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 906

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 961

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5875

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.6250064, upper bound: 0.6240900
time: 7.76 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.6250064, upper bound: 0.6240916
time: 5.41 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -8.8729210, -5.9873095, -8.8933945, -5.9659891, -1.7034864, 1.7022872
1: -16.4740734, -14.1398010, -16.4852562, -14.1285057, -1.5192180, 1.5194397
2: -7.4266195, -5.0024667, -7.4473000, -4.9802032, -1.4843678, 1.4826746
3: -12.9051628, -9.7031965, -12.9179459, -9.6885567, -2.5623722, 2.5606012
4: -3.2553477, -0.9402680, -3.2626805, -0.9345515, -1.7536254, 1.7570467
5: -13.3643789, -10.7405663, -13.3761597, -10.7282848, -1.1630545, 1.1624565
6: -15.2521458, -12.3448124, -15.2657480, -12.3283253, -1.6873951, 1.6843982
7: -7.6436229, -5.0673900, -7.6541471, -5.0573955, -2.1444178, 2.1445637
8: -6.0116138, -3.7082481, -6.0258360, -3.6912565, -1.5263686, 1.5229836
9: 4.6009693, 6.2021012, 4.5875583, 6.2135963, -1.4033217, 1.4052458

Time for backsubstitution: 21.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 5875
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 906

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 961

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5875

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.6250064, upper bound: 0.6332550
time: 4.23 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.6250064, upper bound: 0.6332538
time: 4.66 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -8.9179182, -5.9422245, -8.9110432, -5.9689674, -1.6800165, 1.7327852
1: -16.4937859, -14.1340446, -16.4851494, -14.1488113, -1.4981093, 1.5260367
2: -7.4627028, -4.9772129, -7.4429150, -4.9970646, -1.4571090, 1.5063794
3: -12.9178228, -9.6767569, -12.8922443, -9.6992807, -2.5342560, 2.5705194
4: -3.2502923, -0.9332480, -3.2106323, -0.9493136, -1.7517529, 1.7271557
5: -13.3891087, -10.7256889, -13.3794708, -10.7548695, -1.1436663, 1.1774869
6: -15.2797518, -12.3181543, -15.2706757, -12.3475971, -1.6463685, 1.7011123
7: -7.6546354, -5.0737476, -7.6280584, -5.1147962, -2.1309900, 2.1092701
8: -6.0405860, -3.6840191, -6.0271740, -3.7207136, -1.4974380, 1.5376606
9: 4.5869088, 6.2127342, 4.6266723, 6.1884823, -1.4076681, 1.3605824

Time for backsubstitution: 21.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5875
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 906

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5875

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 961

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6228040, upper bound: 0.6375733
time: 4.51 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6228042, upper bound: 0.6375733
time: 4.51 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -8.9110775, -5.9689674, -8.9221153, -5.9661999, -1.6837516, 1.7401133
1: -16.4851494, -14.1487780, -16.4972076, -14.1284895, -1.5070763, 1.5259633
2: -7.4429388, -4.9970641, -7.4757195, -4.9814000, -1.4567599, 1.5063412
3: -12.8922491, -9.6992779, -12.9349632, -9.6884956, -2.5223341, 2.5920849
4: -3.2106318, -0.9492857, -3.2570643, -0.9273753, -1.7486629, 1.7444510
5: -13.3794956, -10.7548695, -13.3925743, -10.7284775, -1.1571674, 1.1819134
6: -15.2706881, -12.3475828, -15.2833242, -12.3272686, -1.6620626, 1.6964312
7: -7.6280603, -5.1147633, -7.6540084, -5.0481658, -2.1308060, 2.1048489
8: -6.0272007, -3.7207160, -6.0453267, -3.6907883, -1.5125923, 1.5529075
9: 4.6266241, 6.1884871, 4.5887022, 6.2297640, -1.3926692, 1.3721776

Time for backsubstitution: 22.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 5875
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 906

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 1, pos: 961

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.6228040, upper bound: 0.6300737
time: 6.29 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.6300726, upper bound: 0.6300737
time: 5.52 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -8.9221563, -5.9657760, -8.9221582, -5.9657764, -1.6952353, 1.7456093
1: -16.5001640, -14.1283064, -16.5001678, -14.1283054, -1.5234852, 1.5463457
2: -7.4757910, -4.9801378, -7.4757910, -4.9801373, -1.4782147, 1.5335741
3: -12.9350061, -9.6876144, -12.9350061, -9.6876154, -2.5572243, 2.5930347
4: -3.2633624, -0.9272270, -3.2633617, -0.9272258, -1.7702856, 1.7536631
5: -13.3927841, -10.7282486, -13.3927879, -10.7282486, -1.1577110, 1.1907730
6: -15.2848864, -12.3272018, -15.2848883, -12.3272009, -1.6812601, 1.7214642
7: -7.6543293, -5.0435748, -7.6543293, -5.0435719, -2.1692400, 2.1477442
8: -6.0455408, -3.6904950, -6.0455437, -3.6904945, -1.5172801, 1.5613465
9: 4.5871372, 6.2297649, 4.5871363, 6.2297673, -1.4333620, 1.3986626

Time for backsubstitution: 22.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 5875
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 906

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 961

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6228042, upper bound: 0.6392463
time: 4.72 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.6300727, upper bound: 0.6300738
time: 4.62 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 31.83 seconds
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 31.83
Output dim: 9, lower bound: -0.6250064, upper bound: 0.6240900
NS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 31.83
Output dim: 9, lower bound: -0.6250064, upper bound: 0.6240916
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 31.83
Output dim: 9, lower bound: -0.6250064, upper bound: 0.6332550
NS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 31.83
Output dim: 9, lower bound: -0.6250064, upper bound: 0.6332538
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.83
Output dim: 9, lower bound: -0.6228040, upper bound: 0.6375733
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.83
Output dim: 9, lower bound: -0.6228042, upper bound: 0.6375733
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 31.83
Output dim: 9, lower bound: -0.6228040, upper bound: 0.6300737
NS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 31.83
Output dim: 9, lower bound: -0.6300726, upper bound: 0.6300737
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.83
Output dim: 9, lower bound: -0.6228042, upper bound: 0.6392463
NS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 31.83
Output dim: 9, lower bound: -0.6300727, upper bound: 0.6300738

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -8.9179182, -5.9422245, -8.8972034, -5.9690156, -1.6878042, 1.7189260
1: -16.4937859, -14.1340446, -16.4842682, -14.1488314, -1.4983716, 1.5241141
2: -7.4627028, -4.9772129, -7.4383640, -4.9971271, -1.4595351, 1.5018125
3: -12.9178228, -9.6767569, -12.8845100, -9.6995869, -2.5387774, 2.5627909
4: -3.2502923, -0.9332480, -3.2104390, -0.9545116, -1.7464342, 1.7268157
5: -13.3891087, -10.7256889, -13.3717966, -10.7548800, -1.1481552, 1.1697578
6: -15.2797518, -12.3181543, -15.2615261, -12.3479128, -1.6516418, 1.6919146
7: -7.6546354, -5.0737476, -7.6280017, -5.1196332, -2.1260328, 2.1114125
8: -6.0405860, -3.6840191, -6.0171576, -3.7209306, -1.5016699, 1.5274785
9: 4.5869088, 6.2127342, 4.6267023, 6.1819162, -1.4010868, 1.3646708

Time for backsubstitution: 21.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4557
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 1, pos: 4557

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.6228040, upper bound: 0.6300728
time: 4.67 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6228040, upper bound: 0.6375724
time: 5.07 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -8.9179182, -5.9422245, -8.9111519, -5.9443607, -1.6803656, 1.7256618
1: -16.4937859, -14.1340446, -16.4856033, -14.1477814, -1.5000172, 1.5262718
2: -7.4627028, -4.9772129, -7.4431229, -4.9887495, -1.4581099, 1.5058079
3: -12.9178228, -9.6767569, -12.8927526, -9.6847162, -2.5348949, 2.5523148
4: -3.2502923, -0.9332480, -3.2208707, -0.9489336, -1.7459078, 1.7273102
5: -13.3891087, -10.7256889, -13.3794823, -10.7411976, -1.1436901, 1.1730568
6: -15.2797518, -12.3181543, -15.2707844, -12.3302994, -1.6472774, 1.6887321
7: -7.6546354, -5.0737476, -7.6368070, -5.1147413, -2.1215105, 2.1109133
8: -6.0405860, -3.6840191, -6.0273294, -3.7016869, -1.4978142, 1.5376854
9: 4.5869088, 6.2127342, 4.6141500, 6.1885295, -1.3953676, 1.3611345

Time for backsubstitution: 21.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4557
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 4557

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.6228040, upper bound: 0.6300735
time: 6.68 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6228042, upper bound: 0.6375740
time: 4.77 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -8.9146900, -5.9658022, -8.9082823, -5.9658241, -1.6877275, 1.7317374
1: -16.4996738, -14.1283350, -16.4992695, -14.1283588, -1.5226870, 1.5449533
2: -7.4733253, -4.9801750, -7.4712090, -4.9801979, -1.4756694, 1.5289507
3: -12.9308424, -9.6877804, -12.9272690, -9.6879215, -2.5528049, 2.5851555
4: -3.2632580, -0.9300361, -3.2631679, -0.9324467, -1.7648239, 1.7513442
5: -13.3886375, -10.7282524, -13.3850851, -10.7282600, -1.1535130, 1.1830060
6: -15.2799520, -12.3273754, -15.2757206, -12.3275299, -1.6759272, 1.7120686
7: -7.6542902, -5.0461955, -7.6542583, -5.0484438, -2.1642113, 2.1449938
8: -6.0401344, -3.6906128, -6.0354939, -3.6907082, -1.5115643, 1.5510516
9: 4.5871878, 6.2262297, 4.5872183, 6.2231922, -1.4267478, 1.3950572

Time for backsubstitution: 21.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 5798
type: A, layer: 1, pos: 4608
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 906

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 961

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.6250198, upper bound: 0.6319744
time: 4.44 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6250198, upper bound: 0.6392463
time: 4.52 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 30.81 seconds
NS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 30.81
Output dim: 9, lower bound: -0.6228040, upper bound: 0.6300728
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 30.81
Output dim: 9, lower bound: -0.6228040, upper bound: 0.6375724
NS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 30.81
Output dim: 9, lower bound: -0.6228040, upper bound: 0.6300735
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 30.81
Output dim: 9, lower bound: -0.6228042, upper bound: 0.6375740
NS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 30.81
Output dim: 9, lower bound: -0.6250198, upper bound: 0.6319744
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 30.81
Output dim: 9, lower bound: -0.6250198, upper bound: 0.6392463

## BFS NS instance: NS_A2_B1_A2_B1_A2

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

Time for backsubstitution: 21.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5875
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 906

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 5875

### Candidate
type: B, layer: 1, pos: 5798

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6227131, upper bound: 0.6364879
time: 4.65 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6228032, upper bound: 0.6375715
time: 4.51 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -8.9221916, -5.9415979, -8.9111519, -5.9443607, -1.6853619, 1.7258501
1: -16.4976654, -14.1274948, -16.4856033, -14.1477814, -1.5046787, 1.5316834
2: -7.4758930, -4.9731069, -7.4431229, -4.9887495, -1.4696188, 1.5079150
3: -12.9354649, -9.6739368, -12.8927526, -9.6847162, -2.5526114, 2.5544682
4: -3.2673006, -0.9270470, -3.2208707, -0.9489336, -1.7533069, 1.7324867
5: -13.3925571, -10.7148094, -13.3794823, -10.7411976, -1.1470284, 1.1741540
6: -15.2834206, -12.3099766, -15.2707844, -12.3302994, -1.6515555, 1.6975961
7: -7.6628613, -5.0481472, -7.6368070, -5.1147413, -2.1255732, 2.1160030
8: -6.0454674, -3.6717710, -6.0273294, -3.7016869, -1.5059237, 1.5395279
9: 4.5762634, 6.2298045, 4.6141500, 6.1885295, -1.4032869, 1.3667226

Time for backsubstitution: 21.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5875
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 906

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 5875

### Candidate
type: B, layer: 1, pos: 5798

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6227131, upper bound: 0.6364890
time: 4.63 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6228034, upper bound: 0.6375723
time: 4.67 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -8.9222317, -5.9411774, -8.9082823, -5.9658241, -1.6952624, 1.7318146
1: -16.5006161, -14.1273127, -16.4992695, -14.1283588, -1.5234132, 1.5459204
2: -7.4759688, -4.9718428, -7.4712090, -4.9801979, -1.4782724, 1.5311046
3: -12.9355049, -9.6730576, -12.9272690, -9.6879215, -2.5574436, 2.5998373
4: -3.2735956, -0.9268951, -3.2631679, -0.9324467, -1.7750626, 1.7535992
5: -13.3927708, -10.7145844, -13.3850851, -10.7282600, -1.1576390, 1.1830275
6: -15.2849903, -12.3099079, -15.2757206, -12.3275299, -1.6809444, 1.7247975
7: -7.6631818, -5.0435510, -7.6542583, -5.0484438, -2.1730137, 2.1476727
8: -6.0456800, -3.6714787, -6.0354939, -3.6907082, -1.5170360, 1.5534196
9: 4.5747013, 6.2298059, 4.5872183, 6.2231922, -1.4357638, 1.3986368

Time for backsubstitution: 21.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5875
type: B, layer: 1, pos: 5798
type: B, layer: 1, pos: 4608
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 906

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 5875

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5798

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6249283, upper bound: 0.6381566
time: 4.64 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6250190, upper bound: 0.6392437
time: 4.69 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 37.14 seconds
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 37.14
Output dim: 9, lower bound: -0.6227131, upper bound: 0.6364879
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 37.14
Output dim: 9, lower bound: -0.6228032, upper bound: 0.6375715
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 37.14
Output dim: 9, lower bound: -0.6227131, upper bound: 0.6364890
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 37.14
Output dim: 9, lower bound: -0.6228034, upper bound: 0.6375723
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 37.14
Output dim: 9, lower bound: -0.6249283, upper bound: 0.6381566
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 37.14
Output dim: 9, lower bound: -0.6250190, upper bound: 0.6392437

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -8.9215012, -5.9460363, -8.8935099, -5.9778290, -1.6833200, 1.7083917
1: -16.4968052, -14.1301842, -16.4820747, -14.1540451, -1.4970670, 1.5246553
2: -7.4752102, -4.9783506, -7.4346104, -5.0072799, -1.4577899, 1.4911051
3: -12.9286976, -9.6746330, -12.8712683, -9.7036991, -2.5436020, 2.5506544
4: -3.2670517, -0.9398537, -3.2036664, -0.9794774, -1.7228961, 1.7097211
5: -13.3815098, -10.7148685, -13.3501902, -10.7597399, -1.1335425, 1.1493039
6: -15.2771664, -12.3100920, -15.2491217, -12.3506041, -1.6440940, 1.6826334
7: -7.6561251, -5.0484939, -7.6143708, -5.1237025, -2.1105499, 2.1010294
8: -6.0436358, -3.6726446, -6.0134950, -3.7230949, -1.5053649, 1.5231693
9: 4.5791960, 6.2290850, 4.6326752, 6.1790218, -1.3954153, 1.3600903

Time for backsubstitution: 21.62 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 57.75 + 558.27 = 616.01 seconds
