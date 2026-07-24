## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 12)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.9345672972


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (5.7411714, 7.9211383, 5.7411714, 7.9211383, -1.8727684, 1.8727684)
1: (-21.1727886, -18.1166534, -21.1727886, -18.1166534, -2.0088158, 2.0088162)
2: (-5.1316385, -2.9183588, -5.1316385, -2.9183588, -1.7741785, 1.7741785)
3: (-13.6132164, -11.4091663, -13.6132164, -11.4091663, -1.6917539, 1.6917539)
4: (-8.8114805, -6.7004447, -8.8114805, -6.7004447, -1.5649471, 1.5649471)
5: (-7.3504887, -5.3203244, -7.3504887, -5.3203244, -1.4956284, 1.4956284)
6: (-5.1318026, -3.1755292, -5.1318026, -3.1755292, -1.4957285, 1.4957283)
7: (-10.6166391, -7.8068933, -10.6166391, -7.8068933, -2.3836555, 2.3836555)
8: (-3.6183209, -1.3827600, -3.6183209, -1.3827600, -1.7785492, 1.7785492)
9: (-4.4911757, -2.3049521, -4.4911757, -2.3049521, -1.6223621, 1.6223621)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.10 + 33.84 = 56.93 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.9355028, upper bound: 0.9355027

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 500
type: B, layer: 1, pos: 500
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 5844
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 500

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9349290, upper bound: 0.9245491
time: 5.12 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9354960, upper bound: 0.9354944
time: 4.59 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 9.81 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 9.81
Output dim: 0, lower bound: -0.9349290, upper bound: 0.9245491
NS_A2, status: Status.UNKNOWN, split count: 1, time: 9.81
Output dim: 0, lower bound: -0.9354960, upper bound: 0.9354944

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: 5.7450399, 7.9054437, 5.7423644, 7.9167318, -1.8384295, 1.8562503
1: -21.1670704, -18.1340866, -21.1709538, -18.1215534, -1.9360104, 1.9905820
2: -5.1298923, -2.9248238, -5.1310930, -2.9201758, -1.7663884, 1.7673035
3: -13.6087723, -11.4193668, -13.6118631, -11.4120293, -1.6773634, 1.6785192
4: -8.8093596, -6.7023888, -8.8108749, -6.7010140, -1.5615654, 1.5705705
5: -7.3488154, -5.3225927, -7.3499670, -5.3209705, -1.4780521, 1.4897022
6: -5.1240597, -3.1766088, -5.1296096, -3.1758468, -1.4897509, 1.4924207
7: -10.6152105, -7.8105259, -10.6162167, -7.8079157, -2.3814917, 2.3808045
8: -3.6154006, -1.3917155, -3.6174214, -1.3852739, -1.7493773, 1.7686348
9: -4.4816484, -2.3090825, -4.4885025, -2.3063443, -1.6121225, 1.5813518

Time for backsubstitution: 21.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 5844
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 5778

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9313656, upper bound: 0.9245446
time: 4.66 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9349241, upper bound: 0.9245458
time: 4.91 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: 5.6937208, 7.9213634, 5.7411737, 7.9211268, -1.8960414, 1.8716869
1: -21.2176590, -18.1146164, -21.1727886, -18.1166725, -2.0468802, 2.0095072
2: -5.1489029, -2.9164217, -5.1316376, -2.9183693, -1.7918925, 1.7755942
3: -13.6412048, -11.4081907, -13.6132107, -11.4091778, -1.7198009, 1.6916351
4: -8.8143473, -6.6990204, -8.8114777, -6.7004466, -1.5684905, 1.5673923
5: -7.3530002, -5.3186750, -7.3504887, -5.3203249, -1.5031180, 1.4954691
6: -5.1351538, -3.1638434, -5.1317892, -3.1755295, -1.4977922, 1.5077977
7: -10.6287928, -7.8059964, -10.6166382, -7.8068962, -2.3963861, 2.3842487
8: -3.6450679, -1.3819036, -3.6183167, -1.3827770, -1.7958746, 1.7786055
9: -4.4925013, -2.2774243, -4.4911585, -2.3049560, -1.6234293, 1.6463585

Time for backsubstitution: 21.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 500
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 5844
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 5778

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9319327, upper bound: 0.9354897
time: 4.33 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9354911, upper bound: 0.9354895
time: 5.40 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 31.61 seconds
NS_A1_A1, status: Status.VERIFIED, split count: 2, time: 31.61
Output dim: 0, lower bound: -0.9313656, upper bound: 0.9245446
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 31.61
Output dim: 0, lower bound: -0.9349241, upper bound: 0.9245458
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 31.61
Output dim: 0, lower bound: -0.9319327, upper bound: 0.9354897
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 31.61
Output dim: 0, lower bound: -0.9354911, upper bound: 0.9354895

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: 5.7430124, 7.9176078, 5.7423687, 7.9167337, -1.8362417, 1.8685541
1: -21.1692162, -18.1325531, -21.1709557, -18.1215553, -1.9369974, 1.9930472
2: -5.1346111, -2.9226594, -5.1310930, -2.9201791, -1.7722692, 1.7690282
3: -13.6106596, -11.4088974, -13.6118584, -11.4120302, -1.6773343, 1.6889820
4: -8.8235397, -6.7015791, -8.8108730, -6.7010183, -1.5757728, 1.5675669
5: -7.3505850, -5.3118267, -7.3499618, -5.3209705, -1.4791751, 1.5005598
6: -5.1285324, -3.1763632, -5.1296091, -3.1758482, -1.4950786, 1.4922361
7: -10.6206255, -7.8089013, -10.6162176, -7.8079171, -2.3866158, 2.3803711
8: -3.6270940, -1.3904519, -3.6174219, -1.3852794, -1.7611790, 1.7659655
9: -4.4822483, -2.3074417, -4.4885015, -2.3063455, -1.6146464, 1.5804381

Time for backsubstitution: 24.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 5844
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 500

## Relational analysis of NS_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9245438, upper bound: 0.9245458
time: 4.41 seconds

## Relational analysis of NS_A1_A2_B2

### Relational analysis result of NS_A1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9245438, upper bound: 0.9245461
time: 5.64 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: 5.7027564, 7.9209619, 5.7454810, 7.9209399, -1.8869567, 1.8670983
1: -21.2169304, -18.1158276, -21.1724358, -18.1172466, -2.0419807, 2.0047450
2: -5.1479206, -2.9191940, -5.1311707, -2.9196925, -1.7896738, 1.7722754
3: -13.6331654, -11.4085388, -13.6093750, -11.4093399, -1.7112484, 1.6871390
4: -8.8134298, -6.7088675, -8.8110514, -6.7051344, -1.5629621, 1.5570750
5: -7.3450184, -5.3194489, -7.3466845, -5.3206892, -1.4948163, 1.4909725
6: -5.1342554, -3.1648750, -5.1313572, -3.1760230, -1.4958982, 1.5057139
7: -10.6284342, -7.8095570, -10.6164665, -7.8085947, -2.3935385, 2.3794236
8: -3.6441693, -1.3900697, -3.6178799, -1.3866668, -1.7899132, 1.7698908
9: -4.4920998, -2.2781773, -4.4909763, -2.3053126, -1.6182599, 1.6404977

Time for backsubstitution: 23.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 5844
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 500

## Relational analysis of NS_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9209828, upper bound: 0.9349225
time: 4.47 seconds

## Relational analysis of NS_A2_A1_B2

### Relational analysis result of NS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9209828, upper bound: 0.9354910
time: 5.10 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: 5.6912451, 7.9335213, 5.7411799, 7.9211264, -1.8952889, 1.8839841
1: -21.2209282, -18.1130638, -21.1727867, -18.1166725, -2.0469522, 2.0119882
2: -5.1538882, -2.9142590, -5.1316361, -2.9183726, -1.7977042, 1.7773132
3: -13.6435614, -11.3977261, -13.6132069, -11.4091778, -1.7214293, 1.7020960
4: -8.8285437, -6.6981411, -8.8114786, -6.7004547, -1.5830054, 1.5651097
5: -7.3549662, -5.3078837, -7.3504834, -5.3203273, -1.5049448, 1.5065699
6: -5.1397300, -3.1635275, -5.1317892, -3.1755326, -1.5032549, 1.5076885
7: -10.6343260, -7.8043699, -10.6166382, -7.8068991, -2.4006577, 2.3838406
8: -3.6571782, -1.3806415, -3.6183176, -1.3827844, -1.7986746, 1.7759776
9: -4.4930987, -2.2751033, -4.4911594, -2.3049560, -1.6259527, 1.6449451

Time for backsubstitution: 23.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 500
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 5844
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 500

## Relational analysis of NS_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9245438, upper bound: 0.9349224
time: 5.31 seconds

## Relational analysis of NS_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9245438, upper bound: 0.9354896
time: 9.49 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 38.65 seconds
NS_A1_A2_B1, status: Status.VERIFIED, split count: 3, time: 38.65
Output dim: 0, lower bound: -0.9245438, upper bound: 0.9245458
NS_A1_A2_B2, status: Status.VERIFIED, split count: 3, time: 38.65
Output dim: 0, lower bound: -0.9245438, upper bound: 0.9245461
NS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 38.65
Output dim: 0, lower bound: -0.9209828, upper bound: 0.9349225
NS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 38.65
Output dim: 0, lower bound: -0.9209828, upper bound: 0.9354910
NS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 38.65
Output dim: 0, lower bound: -0.9245438, upper bound: 0.9349224
NS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 38.65
Output dim: 0, lower bound: -0.9245438, upper bound: 0.9354896

## BFS NS instance: NS_A2_A1_B1

### Backsubstitution after applying NS history:
0: 5.7028184, 7.9209628, 5.7493553, 7.9052567, -1.8712187, 1.8384814
1: -21.2168751, -18.1158276, -21.1667442, -18.1346626, -2.0245018, 1.9406567
2: -5.1479120, -2.9191961, -5.1294312, -2.9261456, -1.7832193, 1.7662606
3: -13.6331100, -11.4085369, -13.6049223, -11.4195290, -1.6987700, 1.6768723
4: -8.8134279, -6.7088709, -8.8089333, -6.7070770, -1.5673690, 1.5545077
5: -7.3449965, -5.3194528, -7.3450031, -5.3229561, -1.4829431, 1.4748197
6: -5.1342525, -3.1648772, -5.1236315, -3.1771042, -1.4959145, 1.4998193
7: -10.6284294, -7.8095570, -10.6150408, -7.8122249, -2.3909407, 2.3786030
8: -3.6441383, -1.3900723, -3.6149826, -1.3956044, -1.7808132, 1.7440577
9: -4.4921007, -2.2782121, -4.4814634, -2.3094573, -1.5817690, 1.6309712

Time for backsubstitution: 23.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 5844
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5778

## Relational analysis of NS_A2_A1_B1_B1

### Relational analysis result of NS_A2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9209828, upper bound: 0.9313939
time: 4.40 seconds

## Relational analysis of NS_A2_A1_B1_B2

### Relational analysis result of NS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9209828, upper bound: 0.9349225
time: 5.37 seconds

## BFS NS instance: NS_A2_A1_B2

### Backsubstitution after applying NS history:
0: 5.7027564, 7.9209619, 5.6980276, 7.9211740, -1.8872099, 1.8892159
1: -21.2169304, -18.1158276, -21.2173157, -18.1151924, -2.0391464, 2.0389428
2: -5.1479206, -2.9191940, -5.1484389, -2.9177427, -1.7855797, 1.7844815
3: -13.6331654, -11.4085388, -13.6373749, -11.4083557, -1.7124233, 1.7159414
4: -8.8134298, -6.7088675, -8.8139114, -6.7037106, -1.5655117, 1.5607204
5: -7.3450184, -5.3194489, -7.3491964, -5.3190398, -1.4965897, 1.5004015
6: -5.1342554, -3.1648750, -5.1347280, -3.1643348, -1.5010176, 1.5008407
7: -10.6284342, -7.8095570, -10.6286211, -7.8076944, -2.3919582, 2.3899837
8: -3.6441693, -1.3900697, -3.6446357, -1.3857911, -1.7907805, 1.7872210
9: -4.4920998, -2.2781773, -4.4923096, -2.2777839, -1.6418605, 1.6418881

Time for backsubstitution: 23.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 5844
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 5778

## Relational analysis of NS_A2_A1_B2_B1

### Relational analysis result of NS_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9209828, upper bound: 0.9319621
time: 5.40 seconds

## Relational analysis of NS_A2_A1_B2_B2

### Relational analysis result of NS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9209828, upper bound: 0.9354910
time: 5.34 seconds

## BFS NS instance: NS_A2_A2_B1

### Backsubstitution after applying NS history:
0: 5.6913366, 7.9335213, 5.7450428, 7.9054441, -1.8795257, 1.8550320
1: -21.2208481, -18.1130638, -21.1670704, -18.1340885, -2.0294442, 1.9446692
2: -5.1538706, -2.9142599, -5.1298923, -2.9248252, -1.7912421, 1.7714715
3: -13.6434803, -11.3977242, -13.6087656, -11.4193649, -1.7085953, 1.6918736
4: -8.8285398, -6.6981430, -8.8093596, -6.7023964, -1.5873013, 1.5625286
5: -7.3549352, -5.3078871, -7.3488121, -5.3225927, -1.4930530, 1.4904289
6: -5.1397228, -3.1635320, -5.1240587, -3.1766107, -1.5032659, 1.5018287
7: -10.6343164, -7.8043690, -10.6152124, -7.8105273, -2.3982115, 2.3830237
8: -3.6571350, -1.3806415, -3.6154015, -1.3917201, -1.7895603, 1.7507043
9: -4.4930968, -2.2751548, -4.4816518, -2.3090825, -1.5872860, 1.6353989

Time for backsubstitution: 23.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 5844
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 930

## Relational analysis of NS_A2_A2_B1_A1

### Relational analysis result of NS_A2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9234179, upper bound: 0.9318257
time: 4.59 seconds

## Relational analysis of NS_A2_A2_B1_A2

### Relational analysis result of NS_A2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9214458, upper bound: 0.9318256
time: 4.60 seconds

## BFS NS instance: NS_A2_A2_B2

### Backsubstitution after applying NS history:
0: 5.6912451, 7.9335213, 5.6937251, 7.9213619, -1.8955460, 1.8957760
1: -21.2209282, -18.1130638, -21.2176628, -18.1146164, -2.0435934, 2.0461812
2: -5.1538882, -2.9142590, -5.1489029, -2.9164228, -1.7936077, 1.7895164
3: -13.6435614, -11.3977261, -13.6411991, -11.4081917, -1.7226062, 1.7230029
4: -8.8285437, -6.6981411, -8.8143444, -6.6990309, -1.5855594, 1.5687628
5: -7.3549662, -5.3078837, -7.3529954, -5.3186741, -1.5067215, 1.5159941
6: -5.1397300, -3.1635275, -5.1351519, -3.1638446, -1.5083714, 1.5027997
7: -10.6343260, -7.8043699, -10.6287937, -7.8060002, -2.4000559, 2.3943987
8: -3.6571782, -1.3806415, -3.6450698, -1.3819087, -1.7995424, 1.7933006
9: -4.4930987, -2.2751033, -4.4925022, -2.2774234, -1.6462426, 1.6463416

Time for backsubstitution: 23.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 5844
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 930

## Relational analysis of NS_A2_A2_B2_A1

### Relational analysis result of NS_A2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9234179, upper bound: 0.9323957
time: 5.41 seconds

## Relational analysis of NS_A2_A2_B2_A2

### Relational analysis result of NS_A2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9214458, upper bound: 0.9323941
time: 10.43 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 39.67 seconds
NS_A2_A1_B1_B1, status: Status.VERIFIED, split count: 4, time: 39.67
Output dim: 0, lower bound: -0.9209828, upper bound: 0.9313939
NS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 39.67
Output dim: 0, lower bound: -0.9209828, upper bound: 0.9349225
NS_A2_A1_B2_B1, status: Status.VERIFIED, split count: 4, time: 39.67
Output dim: 0, lower bound: -0.9209828, upper bound: 0.9319621
NS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 39.67
Output dim: 0, lower bound: -0.9209828, upper bound: 0.9354910
NS_A2_A2_B1_A1, status: Status.VERIFIED, split count: 4, time: 39.67
Output dim: 0, lower bound: -0.9234179, upper bound: 0.9318257
NS_A2_A2_B1_A2, status: Status.VERIFIED, split count: 4, time: 39.67
Output dim: 0, lower bound: -0.9214458, upper bound: 0.9318256
NS_A2_A2_B2_A1, status: Status.VERIFIED, split count: 4, time: 39.67
Output dim: 0, lower bound: -0.9234179, upper bound: 0.9323957
NS_A2_A2_B2_A2, status: Status.VERIFIED, split count: 4, time: 39.67
Output dim: 0, lower bound: -0.9214458, upper bound: 0.9323941

## BFS NS instance: NS_A2_A1_B1_B2

### Backsubstitution after applying NS history:
0: 5.7027812, 7.9209628, 5.7430124, 7.9176078, -1.8722067, 1.8445792
1: -21.2169075, -18.1158257, -21.1692162, -18.1325531, -2.0257425, 1.9423308
2: -5.1479173, -2.9191933, -5.1346111, -2.9226594, -1.7863464, 1.7717443
3: -13.6331415, -11.4085360, -13.6106596, -11.4088974, -1.7008338, 1.6828213
4: -8.8134308, -6.7088680, -8.8235397, -6.7015791, -1.5728040, 1.5690923
5: -7.3450103, -5.3194509, -7.3505850, -5.3118267, -1.4941502, 1.4805045
6: -5.1342540, -3.1648765, -5.1285324, -3.1763632, -1.4964552, 1.5047896
7: -10.6284332, -7.8095579, -10.6206255, -7.8089013, -2.3943768, 2.3840771
8: -3.6441560, -1.3900721, -3.6270940, -1.3904519, -1.7815862, 1.7562594
9: -4.4921007, -2.2781911, -4.4822483, -2.3074417, -1.5821857, 1.6305912

Time for backsubstitution: 23.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 5844
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 930

## Relational analysis of NS_A2_A1_B1_B2_A1

### Relational analysis result of NS_A2_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9198571, upper bound: 0.9318257
time: 5.43 seconds

## Relational analysis of NS_A2_A1_B1_B2_A2

### Relational analysis result of NS_A2_A1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9178871, upper bound: 0.9318255
time: 4.92 seconds

## BFS NS instance: NS_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: 5.7027564, 7.9209619, 5.6912451, 7.9335213, -1.8881650, 1.8921421
1: -21.2169304, -18.1158276, -21.2209282, -18.1130638, -2.0402136, 2.0402522
2: -5.1479206, -2.9191940, -5.1538882, -2.9142590, -1.7886910, 1.7904291
3: -13.6331654, -11.4085388, -13.6435614, -11.3977261, -1.7142682, 1.7196808
4: -8.8134298, -6.7088675, -8.8285437, -6.6981411, -1.5712914, 1.5756173
5: -7.3450184, -5.3194489, -7.3549662, -5.3078837, -1.5080223, 1.5061898
6: -5.1342554, -3.1648750, -5.1397300, -3.1635275, -1.5016294, 1.5060821
7: -10.6284342, -7.8095570, -10.6343260, -7.8043699, -2.3954458, 2.3955755
8: -3.6441693, -1.3900697, -3.6571782, -1.3806415, -1.7915792, 1.7904367
9: -4.4920998, -2.2781773, -4.4930987, -2.2751033, -1.6426988, 1.6414881

Time for backsubstitution: 23.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 5844
type: A, layer: 1, pos: 5844
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 930

## Relational analysis of NS_A2_A1_B2_B2_A1

### Relational analysis result of NS_A2_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9215177, upper bound: 0.9323942
time: 5.46 seconds

## Relational analysis of NS_A2_A1_B2_B2_A2

### Relational analysis result of NS_A2_A1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9195491, upper bound: 0.9323956
time: 5.65 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 34.83 seconds
NS_A2_A1_B1_B2_A1, status: Status.VERIFIED, split count: 5, time: 34.83
Output dim: 0, lower bound: -0.9198571, upper bound: 0.9318257
NS_A2_A1_B1_B2_A2, status: Status.VERIFIED, split count: 5, time: 34.83
Output dim: 0, lower bound: -0.9178871, upper bound: 0.9318255
NS_A2_A1_B2_B2_A1, status: Status.VERIFIED, split count: 5, time: 34.83
Output dim: 0, lower bound: -0.9215177, upper bound: 0.9323942
NS_A2_A1_B2_B2_A2, status: Status.VERIFIED, split count: 5, time: 34.83
Output dim: 0, lower bound: -0.9195491, upper bound: 0.9323956

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 56.93 + 388.09 = 445.02 seconds
