## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 1.487812356


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-11.0312033, -6.8542709, -11.0312033, -6.8542709, -3.1203337, 3.1203332)
1: (-9.9745626, -6.7860947, -9.9745626, -6.7860947, -2.8023891, 2.8023882)
2: (-4.8420553, -1.6859286, -4.8420553, -1.6859286, -2.8518019, 2.8518019)
3: (-1.7269893, 1.6645042, -1.7269893, 1.6645042, -3.3914933, 3.3914933)
4: (-14.0082026, -10.0282593, -14.0082026, -10.0282593, -3.2965384, 3.2965384)
5: (-8.5575218, -5.0969081, -8.5575218, -5.0969081, -2.1671762, 2.1671762)
6: (-12.7730398, -8.5379305, -12.7730398, -8.5379305, -3.2981486, 3.2981482)
7: (-9.1983805, -5.7004938, -9.1983805, -5.7004938, -2.8508062, 2.8508062)
8: (9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6844096, 2.6844096)
9: (-7.9733386, -3.6992083, -7.9733386, -3.6992083, -2.9997826, 2.9997826)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.42 + 36.86 = 59.28 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -1.4952888, upper bound: 1.4952880

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5845
type: A, layer: 1, pos: 5845
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 6253
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: B, layer: 1, pos: 4630
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 5845

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4952818, upper bound: 1.4930235
time: 11.48 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4952818, upper bound: 1.4952807
time: 5.96 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 17.55 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 17.55
Output dim: 8, lower bound: -1.4952818, upper bound: 1.4930235
NS_B2, status: Status.UNKNOWN, split count: 1, time: 17.55
Output dim: 8, lower bound: -1.4952818, upper bound: 1.4952807

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -11.0299339, -6.8570595, -11.0289764, -6.8591242, -3.1093912, 3.1110620
1: -9.9668093, -6.7874899, -9.9610510, -6.7885375, -2.7918406, 2.7875094
2: -4.8408098, -1.6950128, -4.8398786, -1.7017742, -2.8322558, 2.8390417
3: -1.7237175, 1.6607997, -1.7212691, 1.6580504, -3.3817677, 3.3820689
4: -14.0008450, -10.0292311, -13.9953794, -10.0299644, -3.2829895, 3.2779078
5: -8.5538712, -5.0979543, -8.5511675, -5.0987415, -2.1619563, 2.1599281
6: -12.7711954, -8.5420656, -12.7698059, -8.5451288, -3.2889872, 3.2905388
7: -9.1901073, -5.7015314, -9.1839571, -5.7023139, -2.8390970, 2.8337560
8: 9.6456785, 12.5810013, 9.6516809, 12.5805035, -2.6733260, 2.6667404
9: -7.9725308, -3.7004178, -7.9719172, -3.7013092, -2.9957156, 2.9964938

Time for backsubstitution: 21.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 6253
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 5845
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: A, layer: 1, pos: 4630
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of NS_B1_A1

### Relational analysis result of NS_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4918481, upper bound: 1.4930220
time: 7.35 seconds

## Relational analysis of NS_B1_A2

### Relational analysis result of NS_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4952793, upper bound: 1.4930207
time: 11.83 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -11.0312004, -6.8542795, -11.0377007, -6.8517323, -3.1244545, 3.1243010
1: -9.9745502, -6.7860956, -9.9825945, -6.7680578, -2.8194928, 2.8081851
2: -4.8420515, -1.6859416, -4.8699255, -1.6805713, -2.8484478, 2.8718591
3: -1.7269833, 1.6644989, -1.7423849, 1.6700649, -3.3970482, 3.4068837
4: -14.0081930, -10.0282612, -14.0116749, -10.0041485, -3.3167562, 3.2936316
5: -8.5575132, -5.0969095, -8.5616503, -5.0880117, -2.1761413, 2.1693206
6: -12.7730389, -8.5379362, -12.7881203, -8.5354910, -3.2967787, 3.3128362
7: -9.1983671, -5.7004972, -9.2025146, -5.6786852, -2.8710012, 2.8473611
8: 9.6376400, 12.5816555, 9.6287956, 12.5982714, -2.7011070, 2.6898413
9: -7.9733362, -3.6992133, -7.9776435, -3.6958714, -3.0030966, 3.0041800

Time for backsubstitution: 22.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 6253
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5845
type: B, layer: 1, pos: 4630
type: A, layer: 1, pos: 4630
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4918481, upper bound: 1.4952794
time: 7.97 seconds

## Relational analysis of NS_B2_A2

### Relational analysis result of NS_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4952793, upper bound: 1.4952782
time: 5.78 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 36.25 seconds
NS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 36.25
Output dim: 8, lower bound: -1.4918481, upper bound: 1.4930220
NS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 36.25
Output dim: 8, lower bound: -1.4952793, upper bound: 1.4930207
NS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 36.25
Output dim: 8, lower bound: -1.4918481, upper bound: 1.4952794
NS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 36.25
Output dim: 8, lower bound: -1.4952793, upper bound: 1.4952782

## BFS NS instance: NS_B1_A1

### Backsubstitution after applying NS history:
0: -10.9930849, -6.8987536, -11.0238581, -6.8827534, -3.0520248, 3.0657377
1: -9.9423857, -6.7953248, -9.9572430, -6.7927899, -2.7607174, 2.7727752
2: -4.8268914, -1.7089310, -4.8341970, -1.7054825, -2.8131948, 2.8188915
3: -1.6850262, 1.6204880, -1.6998501, 1.6517205, -3.3367467, 3.3203382
4: -13.9621506, -10.0650969, -13.9887590, -10.0505257, -3.2270451, 3.2369556
5: -8.5343094, -5.1186686, -8.5486135, -5.1099243, -2.1304932, 2.1362739
6: -12.7603817, -8.5537148, -12.7667665, -8.5504627, -3.2686586, 3.2715673
7: -9.1709099, -5.7064886, -9.1751575, -5.7052851, -2.8157606, 2.8140073
8: 9.6634293, 12.5640783, 9.6598644, 12.5777187, -2.6507897, 2.6402068
9: -7.9499083, -3.7145319, -7.9650545, -3.7081106, -2.9651318, 2.9749050

Time for backsubstitution: 22.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 6253
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 5845
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: A, layer: 1, pos: 4630
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of NS_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 848

### Candidate
type: B, layer: 1, pos: 4627

## Relational analysis of NS_B1_A1_B1

### Relational analysis result of NS_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4906116, upper bound: 1.4897173
time: 8.32 seconds

## Relational analysis of NS_B1_A1_B2

### Relational analysis result of NS_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4918420, upper bound: 1.4930143
time: 8.01 seconds

## BFS NS instance: NS_B1_A2

### Backsubstitution after applying NS history:
0: -11.0299282, -6.8570867, -11.0289707, -6.8591433, -3.1093655, 3.0749040
1: -9.9668064, -6.7874942, -9.9610472, -6.7885423, -2.7918310, 2.7851238
2: -4.8408003, -1.6950186, -4.8398724, -1.7017770, -2.8277092, 2.8390293
3: -1.7236836, 1.6607878, -1.7212484, 1.6580453, -3.3817289, 3.3820362
4: -14.0008335, -10.0292549, -13.9953718, -10.0299759, -3.2829609, 3.2684126
5: -8.5538635, -5.0979733, -8.5511637, -5.0987515, -2.1619420, 2.1399312
6: -12.7711906, -8.5420732, -12.7698030, -8.5451374, -3.2778211, 3.2965398
7: -9.1900921, -5.7015352, -9.1839485, -5.7023187, -2.8486061, 2.8215795
8: 9.6456881, 12.5809956, 9.6516876, 12.5805016, -2.6648302, 2.6667275
9: -7.9725184, -3.7004280, -7.9719100, -3.7013183, -2.9988718, 2.9952984

Time for backsubstitution: 22.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 6253
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 5845
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of NS_B1_A2_B1

### Relational analysis result of NS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4946952, upper bound: 1.4880250
time: 6.97 seconds

## Relational analysis of NS_B1_A2_B2

### Relational analysis result of NS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4946952, upper bound: 1.4924373
time: 6.50 seconds

## BFS NS instance: NS_B2_A1

### Backsubstitution after applying NS history:
0: -10.9943514, -6.8959694, -11.0325775, -6.8753538, -3.0670834, 3.0789747
1: -9.9501400, -6.7939305, -9.9787674, -6.7723107, -2.7871857, 2.7934284
2: -4.8281307, -1.6998566, -4.8642464, -1.6842808, -2.8293829, 2.8510120
3: -1.6883099, 1.6241982, -1.7209888, 1.6637623, -3.3520722, 3.3451869
4: -13.9695101, -10.0641222, -14.0050764, -10.0247087, -3.2607937, 3.2526426
5: -8.5379553, -5.1176271, -8.5591040, -5.0991917, -2.1446905, 2.1456511
6: -12.7622185, -8.5495806, -12.7850666, -8.5408173, -3.2764406, 3.2938447
7: -9.1791821, -5.7054539, -9.1937227, -5.6816602, -2.8476400, 2.8275714
8: 9.6553841, 12.5647345, 9.6369743, 12.5954819, -2.6785660, 2.6633096
9: -7.9507079, -3.7133226, -7.9707499, -3.7026708, -2.9725142, 2.9825587

Time for backsubstitution: 22.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 6253
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5845
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of NS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 848

### Candidate
type: B, layer: 1, pos: 4627

## Relational analysis of NS_B2_A1_B1

### Relational analysis result of NS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4906116, upper bound: 1.4919754
time: 11.03 seconds

## Relational analysis of NS_B2_A1_B2

### Relational analysis result of NS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4918420, upper bound: 1.4952728
time: 7.73 seconds

## BFS NS instance: NS_B2_A2

### Backsubstitution after applying NS history:
0: -11.0311947, -6.8543072, -11.0376949, -6.8517466, -3.1244278, 3.0881433
1: -9.9745445, -6.7861032, -9.9825907, -6.7680626, -2.8131418, 2.8058014
2: -4.8420444, -1.6859484, -4.8699179, -1.6805737, -2.8439016, 2.8679566
3: -1.7269542, 1.6644881, -1.7423673, 1.6700577, -3.3970118, 3.4068553
4: -14.0081816, -10.0282831, -14.0116673, -10.0041599, -3.3026552, 3.2841334
5: -8.5575123, -5.0969300, -8.5616493, -5.0880218, -2.1730084, 2.1493232
6: -12.7730360, -8.5379467, -12.7881165, -8.5354939, -3.2856331, 3.3188224
7: -9.1983528, -5.7005000, -9.2025051, -5.6786857, -2.8805122, 2.8352127
8: 9.6376524, 12.5816526, 9.6287994, 12.5982695, -2.6926103, 2.6898265
9: -7.9733253, -3.6992211, -7.9776359, -3.6958797, -3.0062523, 3.0029840

Time for backsubstitution: 22.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 6253
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5845
type: A, layer: 1, pos: 4630
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of NS_B2_A2_B1

### Relational analysis result of NS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4946953, upper bound: 1.4902824
time: 8.15 seconds

## Relational analysis of NS_B2_A2_B2

### Relational analysis result of NS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4946952, upper bound: 1.4946948
time: 5.43 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 36.14 seconds
NS_B1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 36.14
Output dim: 8, lower bound: -1.4906116, upper bound: 1.4897173
NS_B1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 36.14
Output dim: 8, lower bound: -1.4918420, upper bound: 1.4930143
NS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 36.14
Output dim: 8, lower bound: -1.4946952, upper bound: 1.4880250
NS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 36.14
Output dim: 8, lower bound: -1.4946952, upper bound: 1.4924373
NS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 36.14
Output dim: 8, lower bound: -1.4906116, upper bound: 1.4919754
NS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 36.14
Output dim: 8, lower bound: -1.4918420, upper bound: 1.4952728
NS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 36.14
Output dim: 8, lower bound: -1.4946953, upper bound: 1.4902824
NS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 36.14
Output dim: 8, lower bound: -1.4946952, upper bound: 1.4946948

## BFS NS instance: NS_B1_A1_B1

### Backsubstitution after applying NS history:
0: -10.9727221, -6.9076428, -10.9895029, -6.8982134, -3.0063319, 3.0109086
1: -9.9373140, -6.8068404, -9.9485550, -6.8122463, -2.7345304, 2.7508211
2: -4.8240604, -1.7182561, -4.8293657, -1.7211998, -2.7876039, 2.7961025
3: -1.6764755, 1.6171966, -1.6854148, 1.6462164, -3.3226919, 3.3026114
4: -13.9576855, -10.0741997, -13.9811325, -10.0658398, -3.2019625, 3.2143278
5: -8.5279894, -5.1205549, -8.5379381, -5.1131368, -2.1195869, 2.1220162
6: -12.7519398, -8.5626869, -12.7529869, -8.5655174, -3.2441325, 3.2472892
7: -9.1662178, -5.7208118, -9.1670361, -5.7295017, -2.7832289, 2.7879276
8: 9.6751156, 12.5622253, 9.6796188, 12.5745592, -2.6324039, 2.6160815
9: -7.9423242, -3.7179368, -7.9522772, -3.7139187, -2.9499540, 2.9570003

Time for backsubstitution: 22.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6253
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 5845
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 4630
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 848

### Candidate
type: B, layer: 1, pos: 848

### Candidate
type: B, layer: 1, pos: 6253

## Relational analysis of NS_B1_A1_B1_B1

### Relational analysis result of NS_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4881876, upper bound: 1.4891455
time: 6.65 seconds

## Relational analysis of NS_B1_A1_B1_B2

### Relational analysis result of NS_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4906092, upper bound: 1.4897157
time: 6.84 seconds

## BFS NS instance: NS_B1_A1_B2

### Backsubstitution after applying NS history:
0: -10.9930391, -6.8987651, -11.0282393, -6.8498993, -3.0895066, 3.0615406
1: -9.9423771, -6.7953520, -9.9863892, -6.7899356, -2.7600074, 2.7989492
2: -4.8268871, -1.7089541, -4.8499489, -1.7024683, -2.8140125, 2.8290596
3: -1.6850076, 1.6204838, -1.7169545, 1.6617643, -3.3467717, 3.3374381
4: -13.9621429, -10.0651140, -14.0018606, -10.0408506, -3.2323828, 3.2455864
5: -8.5342970, -5.1186719, -8.5529156, -5.0994787, -2.1402869, 2.1397610
6: -12.7603674, -8.5537357, -12.7795258, -8.5427523, -3.2750916, 3.2842665
7: -9.1709051, -5.7065191, -9.2038517, -5.6990800, -2.8170290, 2.8401384
8: 9.6634531, 12.5640764, 9.6437187, 12.5885468, -2.6603289, 2.6513593
9: -7.9498925, -3.7145352, -7.9703860, -3.6946392, -2.9797215, 2.9808683

Time for backsubstitution: 21.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 6253
type: B, layer: 1, pos: 6253
type: A, layer: 1, pos: 5845
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 4630
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 4627

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 848

### Candidate
type: B, layer: 1, pos: 848

### Candidate
type: A, layer: 1, pos: 6253

## Relational analysis of NS_B1_A1_B2_A1

### Relational analysis result of NS_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4912667, upper bound: 1.4905904
time: 8.56 seconds

## Relational analysis of NS_B1_A1_B2_A2

### Relational analysis result of NS_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4918398, upper bound: 1.4930119
time: 6.01 seconds

## BFS NS instance: NS_B1_A2_B1

### Backsubstitution after applying NS history:
0: -11.0293694, -6.8594470, -11.0279531, -6.8634415, -3.1020565, 3.0700150
1: -9.9659615, -6.7911062, -9.9595118, -6.7951250, -2.7832842, 2.7791901
2: -4.8376045, -1.6956447, -4.8340502, -1.7029220, -2.8233361, 2.8324361
3: -1.7184904, 1.6584697, -1.7117727, 1.6538043, -3.3722947, 3.3702424
4: -13.9983244, -10.0363960, -13.9907866, -10.0429983, -3.2678986, 3.2581038
5: -8.5531893, -5.1035872, -8.5499277, -5.1089830, -2.1511517, 2.1332877
6: -12.7705984, -8.5439472, -12.7687225, -8.5485563, -3.2740903, 3.2902474
7: -9.1882076, -5.7022762, -9.1805067, -5.7036667, -2.8392105, 2.8177047
8: 9.6497545, 12.5802231, 9.6591005, 12.5790939, -2.6594963, 2.6585960
9: -7.9712172, -3.7008834, -7.9695325, -3.7021499, -2.9943428, 2.9912486

Time for backsubstitution: 21.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 6253
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 5845
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4630
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 4627

## Relational analysis of NS_B1_A2_B1_A1

### Relational analysis result of NS_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4913909, upper bound: 1.4867877
time: 8.49 seconds

## Relational analysis of NS_B1_A2_B1_A2

### Relational analysis result of NS_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4946889, upper bound: 1.4880190
time: 6.62 seconds

## BFS NS instance: NS_B1_A2_B2

### Backsubstitution after applying NS history:
0: -11.0299149, -6.8570957, -11.0363235, -6.8579192, -3.1041117, 3.0819993
1: -9.9668007, -6.7875490, -9.9828892, -6.7877159, -2.7882085, 2.8024201
2: -4.8407907, -1.6950755, -4.8436956, -1.6918390, -2.8378501, 2.8400779
3: -1.7236018, 1.6607784, -1.7239723, 1.6753906, -3.3989925, 3.3847508
4: -14.0008240, -10.0292931, -14.0147400, -10.0287971, -3.2810898, 3.2842951
5: -8.5538177, -5.0979958, -8.5682621, -5.0967398, -2.1587667, 2.1551385
6: -12.7700081, -8.5420818, -12.7699089, -8.5420494, -3.2863684, 3.2925367
7: -9.1900854, -5.7048969, -9.1886139, -5.7049942, -2.8481140, 2.8275476
8: 9.6458216, 12.5809898, 9.6494131, 12.5929356, -2.6775374, 2.6668949
9: -7.9699783, -3.7004492, -7.9703541, -3.7021894, -2.9912367, 3.0034637

Time for backsubstitution: 22.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 6253
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 5845
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4630
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 4627

## Relational analysis of NS_B1_A2_B2_A1

### Relational analysis result of NS_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4913908, upper bound: 1.4912002
time: 7.49 seconds

## Relational analysis of NS_B1_A2_B2_A2

### Relational analysis result of NS_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4946889, upper bound: 1.4924312
time: 5.52 seconds

## BFS NS instance: NS_B2_A1_B1

### Backsubstitution after applying NS history:
0: -10.9739885, -6.9048600, -10.9982214, -6.8908153, -3.0212173, 3.0241895
1: -9.9450684, -6.8054490, -9.9700832, -6.7917767, -2.7610507, 2.7714510
2: -4.8252983, -1.7091742, -4.8594074, -1.6999673, -2.8037839, 2.8265893
3: -1.6797457, 1.6209090, -1.7064633, 1.6582701, -3.3380158, 3.3273723
4: -13.9650497, -10.0732241, -13.9974775, -10.0400229, -3.2357044, 3.2299833
5: -8.5316353, -5.1195145, -8.5484505, -5.1023998, -2.1337881, 2.1313617
6: -12.7537861, -8.5585537, -12.7711678, -8.5558605, -3.2519236, 3.2695203
7: -9.1744909, -5.7197771, -9.1855774, -5.7058868, -2.8150883, 2.8014717
8: 9.6670618, 12.5628805, 9.6566706, 12.5923204, -2.6601882, 2.6391335
9: -7.9431257, -3.7167268, -7.9579716, -3.7084589, -2.9573298, 2.9646516

Time for backsubstitution: 22.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6253
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5845
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 4630
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 848

### Candidate
type: B, layer: 1, pos: 848

### Candidate
type: B, layer: 1, pos: 6253

## Relational analysis of NS_B2_A1_B1_B1

### Relational analysis result of NS_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4881877, upper bound: 1.4914053
time: 6.59 seconds

## Relational analysis of NS_B2_A1_B1_B2

### Relational analysis result of NS_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4906092, upper bound: 1.4919732
time: 7.00 seconds

## BFS NS instance: NS_B2_A1_B2

### Backsubstitution after applying NS history:
0: -10.9943085, -6.8959799, -11.0369663, -6.8424830, -3.0995746, 3.0747828
1: -9.9501314, -6.7939582, -10.0079117, -6.7694411, -2.7865591, 2.8198221
2: -4.8281250, -1.6998796, -4.8800120, -1.6812521, -2.8301649, 2.8536656
3: -1.6882901, 1.6241958, -1.7380760, 1.6737659, -3.3620560, 3.3622718
4: -13.9695063, -10.0641394, -14.0181665, -10.0150280, -3.2661219, 3.2613072
5: -8.5379391, -5.1176319, -8.5634212, -5.0887527, -2.1544819, 2.1491361
6: -12.7622070, -8.5496016, -12.7977962, -8.5331230, -3.2828679, 3.3065205
7: -9.1791754, -5.7054873, -9.2223959, -5.6754608, -2.8488970, 2.8537560
8: 9.6554079, 12.5647297, 9.6207848, 12.6063213, -2.6872926, 2.6744294
9: -7.9506912, -3.7133317, -7.9760904, -3.6892066, -2.9870949, 2.9885321

Time for backsubstitution: 22.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6253
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5845
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 4630
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 4627

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 848

### Candidate
type: B, layer: 1, pos: 848

### Candidate
type: B, layer: 1, pos: 6253

## Relational analysis of NS_B2_A1_B2_B1

### Relational analysis result of NS_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4894152, upper bound: 1.4946979
time: 6.77 seconds

## Relational analysis of NS_B2_A1_B2_B2

### Relational analysis result of NS_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4918396, upper bound: 1.4952714
time: 8.81 seconds

## BFS NS instance: NS_B2_A2_B1

### Backsubstitution after applying NS history:
0: -11.0306349, -6.8566651, -11.0366774, -6.8560457, -3.1171212, 3.0832543
1: -9.9737034, -6.7897129, -9.9810390, -6.7746458, -2.8045754, 2.7998729
2: -4.8388462, -1.6865721, -4.8640976, -1.6817203, -2.8395271, 2.8613782
3: -1.7217662, 1.6621706, -1.7329118, 1.6658355, -3.3876019, 3.3950825
4: -14.0056753, -10.0354252, -14.0071030, -10.0171814, -3.2875156, 3.2737985
5: -8.5568352, -5.1025400, -8.5604162, -5.0982451, -2.1621675, 2.1426790
6: -12.7724400, -8.5398169, -12.7870369, -8.5389071, -3.2818851, 3.3125248
7: -9.1964703, -5.7012396, -9.1990709, -5.6800356, -2.8711138, 2.8313098
8: 9.6417198, 12.5808811, 9.6362152, 12.5968571, -2.6872673, 2.6816990
9: -7.9720211, -3.6996827, -7.9752445, -3.6967168, -3.0017271, 2.9989257

Time for backsubstitution: 22.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 6253
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 5845
type: A, layer: 1, pos: 4630
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 4627

## Relational analysis of NS_B2_A2_B1_A1

### Relational analysis result of NS_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4913909, upper bound: 1.4890437
time: 14.20 seconds

## Relational analysis of NS_B2_A2_B1_A2

### Relational analysis result of NS_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4946889, upper bound: 1.4902769
time: 26.36 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 59.28 + 541.93 = 601.21 seconds
