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
execution time: IAR + RelationalAnalysis = 22.87 + 33.92 = 56.79 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.9355028, upper bound: 0.9355027

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 5844
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 95

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 500

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9349290, upper bound: 0.9245491
time: 5.23 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9354960, upper bound: 0.9354944
time: 4.55 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 9.87 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 9.87
Output dim: 0, lower bound: -0.9349290, upper bound: 0.9245491
NS_A2, status: Status.UNKNOWN, split count: 1, time: 9.87
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

Time for backsubstitution: 21.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 5844
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 95

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 500

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9245478, upper bound: 0.9245474
time: 4.60 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9245478, upper bound: 0.9245476
time: 9.22 seconds

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

Time for backsubstitution: 21.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 5844
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 95

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 500

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9245478, upper bound: 0.9349287
time: 4.81 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9245478, upper bound: 0.9349290
time: 7.47 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 33.40 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 33.40
Output dim: 0, lower bound: -0.9245478, upper bound: 0.9245474
NS_A1_B2, status: Status.VERIFIED, split count: 2, time: 33.40
Output dim: 0, lower bound: -0.9245478, upper bound: 0.9245476
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 33.40
Output dim: 0, lower bound: -0.9245478, upper bound: 0.9349287
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 33.40
Output dim: 0, lower bound: -0.9245478, upper bound: 0.9349290

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: 5.6937451, 7.9213634, 5.7450399, 7.9054437, -1.8803339, 1.8430529
1: -21.2176380, -18.1146164, -21.1670704, -18.1340866, -2.0294356, 1.9431567
2: -5.1489000, -2.9164219, -5.1298923, -2.9248238, -1.7854476, 1.7702336
3: -13.6411829, -11.4081917, -13.6087723, -11.4193668, -1.7073517, 1.6813788
4: -8.8143454, -6.6990209, -8.8093596, -6.7023888, -1.5729198, 1.5648255
5: -7.3529911, -5.3186741, -7.3488154, -5.3225927, -1.4912658, 1.4792747
6: -5.1351519, -3.1638441, -5.1240597, -3.1766088, -1.4978094, 1.5020704
7: -10.6287918, -7.8059978, -10.6152105, -7.8105259, -2.3938560, 2.3834333
8: -3.6450565, -1.3819025, -3.6154006, -1.3917155, -1.7867947, 1.7527533
9: -4.4925003, -2.2774386, -4.4816484, -2.3090825, -1.5853786, 1.6368537

Time for backsubstitution: 21.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5844
type: A, layer: 1, pos: 95

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 5778

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9209828, upper bound: 0.9349226
time: 4.45 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9245438, upper bound: 0.9349224
time: 4.55 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: 5.6937208, 7.9213634, 5.6937208, 7.9213634, -1.8962975, 1.8949842
1: -21.2176590, -18.1146164, -21.2176590, -18.1146164, -2.0437012, 2.0437016
2: -5.1489029, -2.9164217, -5.1489029, -2.9164217, -1.7877955, 1.7877960
3: -13.6412048, -11.4081907, -13.6412048, -11.4081907, -1.7209768, 1.7214065
4: -8.8143473, -6.6990204, -8.8143473, -6.6990204, -1.5710444, 1.5710444
5: -7.3530002, -5.3186750, -7.3530002, -5.3186750, -1.5048938, 1.5048938
6: -5.1351538, -3.1638434, -5.1351538, -3.1638434, -1.5029082, 1.5029080
7: -10.6287928, -7.8059964, -10.6287928, -7.8059964, -2.3948078, 2.3948078
8: -3.6450679, -1.3819036, -3.6450679, -1.3819036, -1.7967429, 1.7959356
9: -4.4925013, -2.2774243, -4.4925013, -2.2774243, -1.6474919, 1.6477532

Time for backsubstitution: 21.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5844
type: A, layer: 1, pos: 95

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 5778

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9209828, upper bound: 0.9354910
time: 5.28 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9245438, upper bound: 0.9354896
time: 9.48 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 35.95 seconds
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 35.95
Output dim: 0, lower bound: -0.9209828, upper bound: 0.9349226
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 35.95
Output dim: 0, lower bound: -0.9245438, upper bound: 0.9349224
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 35.95
Output dim: 0, lower bound: -0.9209828, upper bound: 0.9354910
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 35.95
Output dim: 0, lower bound: -0.9245438, upper bound: 0.9354896

## BFS NS instance: NS_A2_B1_A1

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

Time for backsubstitution: 21.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 5844
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 95

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 5778

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9209843, upper bound: 0.9313637
time: 4.54 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9209843, upper bound: 0.9349225
time: 4.84 seconds

## BFS NS instance: NS_A2_B1_A2

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

Time for backsubstitution: 21.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 5844
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 95

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 5778

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9245447, upper bound: 0.9313637
time: 4.56 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9245447, upper bound: 0.9349223
time: 5.49 seconds

## BFS NS instance: NS_A2_B2_A1

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

Time for backsubstitution: 21.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5844
type: B, layer: 1, pos: 95

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 5778

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9226443, upper bound: 0.9319325
time: 4.38 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9226443, upper bound: 0.9354911
time: 4.42 seconds

## BFS NS instance: NS_A2_B2_A2

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

Time for backsubstitution: 20.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 466
type: B, layer: 1, pos: 5844
type: B, layer: 1, pos: 95

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 5778

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9262045, upper bound: 0.9319324
time: 4.52 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9262045, upper bound: 0.9354911
time: 4.34 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 29.73 seconds
NS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 29.73
Output dim: 0, lower bound: -0.9209843, upper bound: 0.9313637
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.73
Output dim: 0, lower bound: -0.9209843, upper bound: 0.9349225
NS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 29.73
Output dim: 0, lower bound: -0.9245447, upper bound: 0.9313637
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.73
Output dim: 0, lower bound: -0.9245447, upper bound: 0.9349223
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 29.73
Output dim: 0, lower bound: -0.9226443, upper bound: 0.9319325
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.73
Output dim: 0, lower bound: -0.9226443, upper bound: 0.9354911
NS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 29.73
Output dim: 0, lower bound: -0.9262045, upper bound: 0.9319324
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.73
Output dim: 0, lower bound: -0.9262045, upper bound: 0.9354911

## BFS NS instance: NS_A2_B1_A1_B2

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

Time for backsubstitution: 21.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5844
type: A, layer: 1, pos: 95

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 930

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9198571, upper bound: 0.9318257
time: 5.06 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9178871, upper bound: 0.9318255
time: 4.62 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 5.6913347, 7.9335213, 5.7430124, 7.9176078, -1.8803244, 1.8447270
1: -21.2208519, -18.1130638, -21.1692162, -18.1325531, -2.0334425, 1.9462423
2: -5.1538706, -2.9142599, -5.1346111, -2.9226594, -1.7931232, 1.7779517
3: -13.6434803, -11.3977242, -13.6106596, -11.4088974, -1.7107291, 1.6876378
4: -8.8285408, -6.6981449, -8.8235397, -6.7015791, -1.5765681, 1.5697742
5: -7.3549342, -5.3078871, -7.3505850, -5.3118267, -1.5035787, 1.4899607
6: -5.1397228, -3.1635320, -5.1285324, -3.1763632, -1.5035462, 1.5077405
7: -10.6343155, -7.8043685, -10.6206255, -7.8089013, -2.3943214, 2.3838482
8: -3.6571360, -1.3806405, -3.6270940, -1.3904519, -1.7901654, 1.7570429
9: -4.4930968, -2.2751558, -4.4822483, -2.3074417, -1.5883727, 1.6391110

Time for backsubstitution: 20.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5844
type: A, layer: 1, pos: 95

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 930

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9234175, upper bound: 0.9282664
time: 4.84 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9214473, upper bound: 0.9282663
time: 5.18 seconds

## BFS NS instance: NS_A2_B2_A1_B2

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

Time for backsubstitution: 20.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5844
type: A, layer: 1, pos: 95

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 930

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9215177, upper bound: 0.9323942
time: 5.21 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9195491, upper bound: 0.9323943
time: 5.56 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 5.6912451, 7.9335213, 5.6912451, 7.9335213, -1.8963375, 1.8983357
1: -21.2209282, -18.1130638, -21.2209282, -18.1130638, -2.0480485, 2.0480485
2: -5.1538882, -2.9142590, -5.1538882, -2.9142590, -1.7954841, 1.7954845
3: -13.6435614, -11.3977261, -13.6435614, -11.3977261, -1.7242026, 1.7260122
4: -8.8285437, -6.6981411, -8.8285437, -6.6981411, -1.5763164, 1.5763164
5: -7.3549662, -5.3078837, -7.3549662, -5.3078837, -1.5174894, 1.5174894
6: -5.1397300, -3.1635275, -5.1397300, -3.1635275, -1.5087261, 1.5087261
7: -10.6343260, -7.8043699, -10.6343260, -7.8043699, -2.3953085, 2.3953085
8: -3.6571782, -1.3806415, -3.6571782, -1.3806415, -1.8001895, 1.7961006
9: -4.4930987, -2.2751033, -4.4930987, -2.2751033, -1.6478972, 1.6500502

Time for backsubstitution: 21.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5844
type: A, layer: 1, pos: 95

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 930

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9250779, upper bound: 0.9288372
time: 4.74 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9231094, upper bound: 0.9288373
time: 5.21 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 31.56 seconds
NS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 31.56
Output dim: 0, lower bound: -0.9198571, upper bound: 0.9318257
NS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 31.56
Output dim: 0, lower bound: -0.9178871, upper bound: 0.9318255
NS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 31.56
Output dim: 0, lower bound: -0.9234175, upper bound: 0.9282664
NS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 31.56
Output dim: 0, lower bound: -0.9214473, upper bound: 0.9282663
NS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 31.56
Output dim: 0, lower bound: -0.9215177, upper bound: 0.9323942
NS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 31.56
Output dim: 0, lower bound: -0.9195491, upper bound: 0.9323943
NS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 31.56
Output dim: 0, lower bound: -0.9250779, upper bound: 0.9288372
NS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 31.56
Output dim: 0, lower bound: -0.9231094, upper bound: 0.9288373

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 56.79 + 391.09 = 447.88 seconds
